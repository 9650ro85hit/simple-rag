
import json
import shutil
import sqlite3
from functools import lru_cache
from pathlib import Path
from uuid import uuid4


import asyncio
import json
from fastapi.responses import StreamingResponse



from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.callbacks import UsageMetadataCallbackHandler


from fastapi.responses import Response
import csv
from datetime import date, datetime


from llm_call import llm
from rag_graph import build_rag_graph
from rag_storage import (
    BASE_DIR,
    RAGS_DIR,
    create_rag,
    load_catalog,
    load_vector_store,
)
from feedback_storage import save_feedback, get_feedback


app = FastAPI(title="RAG API")

CHATS_FILE = BASE_DIR / "chats.json"
DB_FILE = BASE_DIR / "chat_memory.sqlite"
CATALOG_FILE = BASE_DIR / "rags.json"


# ---------- JSON storage ----------

def load_json(path):
    if not path.exists() or path.stat().st_size == 0:
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    with temp.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.flush()

    temp.replace(path)


def load_chats():
    return load_json(CHATS_FILE)


def save_chats(chats):
    save_json(CHATS_FILE, chats)


# ---------- SQLite checkpoint ----------

@lru_cache(maxsize=1)
def get_checkpointer():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(DB_FILE),
        check_same_thread=False,
        timeout=30,
    )

    saver = SqliteSaver(conn)
    saver.setup()

    return saver


def delete_thread_checkpoints(thread_id):
    saver = get_checkpointer()
    conn = saver.conn

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    with conn:
        for table in ("writes", "checkpoints", "checkpoint_blobs", "blobs"):
            if table not in tables:
                continue

            columns = {
                row[1]
                for row in conn.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }

            if "thread_id" in columns:
                conn.execute(
                    f"DELETE FROM {table} WHERE thread_id = ?",
                    (thread_id,),
                )


# ---------- Graphs ----------

@lru_cache(maxsize=20)
def get_graph(rag_id):
    return build_rag_graph(
        vector_store=load_vector_store(rag_id, show_ui=False),
        checkpointer=get_checkpointer(),
    )


def call_normal_llm(state):
    messages = state["messages"]
    query = messages[-1].content if messages else ""

    history = [
        {
            "role": message.type,
            "content": message.content,
        }
        for message in messages[:-1]
    ]

    response = llm.invoke(messages)

    return {
        "messages": [
            AIMessage(
                content=response.content,
                additional_kwargs={
                    "answer_source": "LLM knowledge",
                    "feedback_query": query,
                    "feedback_context": history,
                },
                response_metadata=response.response_metadata,
                usage_metadata=response.usage_metadata,
            )
        ]
    }


@lru_cache(maxsize=1)
def get_general_graph():
    builder = StateGraph(MessagesState)

    builder.add_node("llm", call_normal_llm)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)

    return builder.compile(checkpointer=get_checkpointer())


# ---------- Project upload adapter ----------

class MemoryUpload:
    """Adapts FastAPI uploads to the interface expected by create_rag()."""

    def __init__(self, filename, content):
        self.name = filename
        self._content = content

    def getvalue(self):
        return self._content


# @app.post("/api/projects")
# async def create_project(
#     name: str = Form(...),
#     files: list[UploadFile] = File(...),
# ):
#     if not name.strip():
#         raise HTTPException(400, "Project name cannot be empty.")

#     if not files:
#         raise HTTPException(400, "Upload at least one document.")

#     uploaded_files = []

#     for file in files:
#         content = await file.read()

#         if not content:
#             raise HTTPException(400, f"{file.filename} is empty.")

#         uploaded_files.append(
#             MemoryUpload(file.filename, content)
#         )

#     try:
#         rag_id = create_rag(
#             name=name.strip(),
#             uploaded_files=uploaded_files,
#             show_ui=False,
#         )

#         return {
#             "rag_id": rag_id,
#             "name": name.strip(),
#         }

#     except Exception as exc:
#         raise HTTPException(500, str(exc))


@app.post("/api/projects/stream")
async def create_project_stream(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    if not name.strip():
        raise HTTPException(400, "Project name cannot be empty.")

    if not files:
        raise HTTPException(400, "Upload at least one document.")

    uploaded_files = []

    for file in files:
        content = await file.read()

        if not content:
            raise HTTPException(400, f"{file.filename} is empty.")

        uploaded_files.append(
            MemoryUpload(file.filename, content)
        )

    async def event_generator():
        loop = asyncio.get_running_loop()
        progress_queue = asyncio.Queue()

        def progress_callback(data):
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                data
            )

        async def run_creation():
            try:
                rag_id = await asyncio.to_thread(
                    create_rag,
                    name.strip(),
                    uploaded_files,
                    False,
                    progress_callback
                )

                await progress_queue.put({
                    "type": "done",
                    "rag_id": rag_id,
                    "name": name.strip()
                })

            except Exception as exc:
                await progress_queue.put({
                    "type": "error",
                    "message": str(exc)
                })

        task = asyncio.create_task(run_creation())

  
        while True:
            try:
                event = await asyncio.wait_for(
                    progress_queue.get(),
                    timeout=15
                )

                yield f"data: {json.dumps(event)}\n\n"

                if event.get("type") in ("done", "error"):
                    break

            except asyncio.TimeoutError:
                # Keep the SSE connection alive during long operations
                yield ": keep-alive\n\n"

        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )



@app.get("/api/projects")
def list_projects():
    return load_catalog()


@app.patch("/api/projects/{rag_id}")
def rename_project(rag_id: str, body: dict):
    name = str(body.get("name", "")).strip()

    if not name:
        raise HTTPException(400, "Project name cannot be empty.")

    catalog = load_catalog()

    if rag_id not in catalog:
        raise HTTPException(404, "Project not found.")

    catalog[rag_id]["name"] = name
    save_json(CATALOG_FILE, catalog)

    return {"success": True, "name": name}


@app.delete("/api/projects/{rag_id}")
def delete_project(rag_id: str):
    catalog = load_catalog()

    if rag_id not in catalog:
        raise HTTPException(404, "Project not found.")

    chats = load_chats()

    project_chat_ids = [
        chat_id
        for chat_id, chat in chats.items()
        if chat.get("rag_id") == rag_id
        and chat.get("mode", "project") == "project"
    ]

    for chat_id in project_chat_ids:
        delete_thread_checkpoints(chat_id)
        chats.pop(chat_id, None)

    save_chats(chats)

    project_dir = (RAGS_DIR / rag_id).resolve()

    if project_dir.parent != RAGS_DIR.resolve():
        raise HTTPException(400, "Invalid project storage path.")

    get_graph.cache_clear()

    if project_dir.exists():
        shutil.rmtree(project_dir)

    catalog.pop(rag_id, None)
    save_json(CATALOG_FILE, catalog)

    return {"success": True}


# ---------- Chat management ----------

class CreateChatRequest(BaseModel):
    mode: str
    rag_id: str | None = None



@app.get("/api/chats")
def list_chats():
    chats = load_chats()
    saver = get_checkpointer()

    for chat_id, chat in chats.items():
        config = {
            "configurable": {
                "thread_id": chat_id
            }
        }

        checkpoint = saver.get_tuple(config)

        messages = []

        if checkpoint:
            channel_values = checkpoint.checkpoint.get(
                "channel_values", {}
            )
            messages = channel_values.get("messages", [])

        # chat["message_count"] = len(messages)
        chat["message_count"] = sum(
            1 for message in messages
            if getattr(message, "type", None) == "human"
        )

    return chats



@app.post("/api/chats")
def create_chat(body: CreateChatRequest):
    if body.mode not in ("general", "project"):
        raise HTTPException(400, "Invalid chat mode.")

    if body.mode == "project":
        if not body.rag_id or body.rag_id not in load_catalog():
            raise HTTPException(404, "Project not found.")

    chat_id = str(uuid4())
    chats = load_chats()

    chats[chat_id] = {
        "title": "New Chat",
        "mode": body.mode,
        "rag_id": body.rag_id,
    }

    save_chats(chats)

    return {"chat_id": chat_id, **chats[chat_id]}


@app.patch("/api/chats/{chat_id}")
def rename_chat(chat_id: str, body: dict):
    name = str(body.get("title", "")).strip()

    if not name:
        raise HTTPException(400, "Chat name cannot be empty.")

    chats = load_chats()

    if chat_id not in chats:
        raise HTTPException(404, "Chat not found.")

    chats[chat_id]["title"] = name
    save_chats(chats)

    return {"success": True, "title": name}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    chats = load_chats()

    if chat_id not in chats:
        raise HTTPException(404, "Chat not found.")

    delete_thread_checkpoints(chat_id)

    chats.pop(chat_id)
    save_chats(chats)

    return {"success": True}


@app.get("/api/chats/{chat_id}/messages")
def get_chat_messages(chat_id: str):
    chats = load_chats()
    chat = chats.get(chat_id)

    if not chat:
        raise HTTPException(404, "Chat not found.")

    mode = chat.get("mode", "general")
    rag_id = chat.get("rag_id")

    # Get the correct graph
    if mode == "project":
        if not rag_id or rag_id not in load_catalog():
            raise HTTPException(404, "Project not found.")

        graph = get_graph(rag_id)
    else:
        graph = get_general_graph()

    # Retrieve saved conversation state from SQLite
    config = {
        "configurable": {
            "thread_id": chat_id
        }
    }

    state = graph.get_state(config)
    messages = state.values.get("messages", [])

    result = []

    for message in messages:
        role = {
            "human": "user",
            "ai": "assistant",
            "system": "system"
        }.get(message.type, message.type)

        additional = getattr(message, "additional_kwargs", {})
        if message.type == "ai":
            print(
                "HISTORY CHECK:",
                message.id,
                "Chunks:",
                len(additional.get("retrieved_chunks", []))
            )

        response_id = message.id if message.type == "ai" else None
        saved_feedback = get_feedback(response_id) if response_id else None

        result.append({
            "id": message.id,
            "role": role,
            "text": (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            ),
            "answer_source": additional.get("answer_source"),
            "usage": additional.get("usage"),
            "context_usage": additional.get("context_usage"),
            "feedback_query": additional.get("feedback_query"),
            "feedback_context": additional.get("feedback_context"),
            "response_id": response_id,
            "retrieved_chunks": additional.get("retrieved_chunks", []),
            "feedback": (
                "up" if saved_feedback == "good"
                else "down" if saved_feedback == "bad"
                else None
            )
        })

    return {
        "chat_id": chat_id,
        "messages": result
    }

# ---------- Chat response streaming ----------

class ChatRequest(BaseModel):
    chat_id: str
    query: str


def make_event(event_type, **data):
    return (
        "data: "
        + json.dumps({"type": event_type, **data}, ensure_ascii=False)
        + "\n\n"
    )



# @app.post("/api/chat/stream")
# def stream_chat(body: ChatRequest):
#     chats = load_chats()
#     chat = chats.get(body.chat_id)

#     if not chat:
#         raise HTTPException(404, "Chat not found.")

#     query = body.query.strip()

#     if not query:
#         raise HTTPException(400, "Question cannot be empty.")

#     mode = chat.get("mode", "general")
#     rag_id = chat.get("rag_id")

#     if mode == "project":
#         if not rag_id or rag_id not in load_catalog():
#             raise HTTPException(404, "Project not found.")

#         graph = get_graph(rag_id)
#     else:
#         graph = get_general_graph()

#     def generate():
#         callback = UsageMetadataCallbackHandler()

#         config = {
#             "configurable": {"thread_id": body.chat_id},
#             "callbacks": [callback],
#         }

#         # Track text already streamed to the frontend.
#         streamed_text = ""

#         def make_token_event(content):
#             nonlocal streamed_text

#             if not isinstance(content, str) or not content:
#                 return None

#             # Handle cumulative output.
#             if content.startswith(streamed_text):
#                 content = content[len(streamed_text):]

#             # Ignore a repeated answer already streamed.
#             elif streamed_text.startswith(content):
#                 return None

#             if not content:
#                 return None

#             streamed_text += content
#             return make_event("token", content=content)

#         try:
#             graph_input = {
#                 "messages": [HumanMessage(content=query)]
#             }

#             # Project / RAG streaming
#             if mode == "project":
#                 graph_input["query"] = query

#                 for event in graph.stream(
#                     graph_input,
#                     config=config,
#                     stream_mode="custom",
#                 ):
#                     if (
#                         isinstance(event, dict)
#                         and event.get("type") == "answer_token"
#                     ):
#                         token_event = make_token_event(
#                             event.get("content", "")
#                         )

#                         if token_event:
#                             yield token_event

#             # General chatbot streaming
#             else:
#                 for message, metadata in graph.stream(
#                     graph_input,
#                     config=config,
#                     stream_mode="messages",
#                 ):
#                     if metadata.get("langgraph_node") == "llm":
#                         content = getattr(message, "content", "")

#                         token_event = make_token_event(content)

#                         if token_event:
#                             yield token_event

#             # Get final graph state
#             state = graph.get_state(config)
#             messages = state.values.get("messages", [])

#             last_message = messages[-1] if messages else None

#             if last_message:
#                 print(
#                     "FINAL MESSAGE USAGE:",
#                     last_message.usage_metadata
#                 )

#             additional = (
#                 last_message.additional_kwargs
#                 if last_message else {}
#             )

#             # Calculate cumulative API usage for this request.
#             usage = {
#                 "input_tokens": 0,
#                 "output_tokens": 0,
#                 "total_tokens": 0,
#             }

#             for model_usage in callback.usage_metadata.values():
#                 if not isinstance(model_usage, dict):
#                     continue

#                 input_tokens = model_usage.get("input_tokens", 0)
#                 output_tokens = model_usage.get("output_tokens", 0)

#                 usage["input_tokens"] += input_tokens
#                 usage["output_tokens"] += output_tokens
#                 usage["total_tokens"] += model_usage.get(
#                     "total_tokens",
#                     input_tokens + output_tokens,
#                 )

#             # Save usage metadata in the final AI message.
#             if last_message and last_message.type == "ai":
#                 last_message.additional_kwargs["usage"] = usage

#                 graph.update_state(
#                     config,
#                     {"messages": [last_message]}
#                 )

#             # Context usage from the final AI message.
#             # Context usage from the final AI message.
#             final_usage = (
#                 last_message.usage_metadata
#                 if last_message and last_message.usage_metadata
#                 else {}
#             )

#             # Fall back to callback usage when message metadata is unavailable.
#             context_usage = {
#                 "input_tokens": final_usage.get(
#                     "input_tokens", usage["input_tokens"]
#                 ),
#                 "output_tokens": final_usage.get(
#                     "output_tokens", usage["output_tokens"]
#                 ),
#                 "total_tokens": final_usage.get(
#                     "total_tokens", usage["total_tokens"]
#                 ),
#             }

#             # Send final metadata (not answer text).
#             yield make_event(
#                 "done",
#                 response_id=last_message.id if last_message else None,
#                 answer_source=additional.get(
#                     "answer_source",
#                     "LLM knowledge",
#                 ),
#                 feedback_query=additional.get(
#                     "feedback_query",
#                     query
#                 ),
#                 feedback_context=additional.get(
#                     "feedback_context",
#                     []
#                 ),
#                 usage=usage,
#                 context_usage=context_usage,
#             )

#         except Exception as exc:
#             yield make_event("error", message=str(exc))

#     return StreamingResponse(
#         generate(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "X-Accel-Buffering": "no",
#         },
#     )




@app.post("/api/chat/stream")
def stream_chat(body: ChatRequest):
    chats = load_chats()
    chat = chats.get(body.chat_id)

    if not chat:
        raise HTTPException(404, "Chat not found.")

    query = body.query.strip()

    if not query:
        raise HTTPException(400, "Question cannot be empty.")

    mode = chat.get("mode", "general")
    rag_id = chat.get("rag_id")

    if mode == "project":
        if not rag_id or rag_id not in load_catalog():
            raise HTTPException(404, "Project not found.")

        graph = get_graph(rag_id)
    else:
        graph = get_general_graph()

    def content_to_text(content):
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
                and isinstance(block.get("text"), str)
            )

        return ""

    def generate():
        callback = UsageMetadataCallbackHandler()

        config = {
            "configurable": {"thread_id": body.chat_id},
            "callbacks": [callback],
        }

        # Collect chunks only as a fallback.
        streamed_parts = []

        try:
            graph_input = {
                "messages": [HumanMessage(content=query)]
            }

            if mode == "project":
                graph_input["query"] = query

                for event in graph.stream(
                    graph_input,
                    config=config,
                    stream_mode="custom",
                ):
                    if (
                        isinstance(event, dict)
                        and event.get("type") == "answer_token"
                    ):
                        content = event.get("content", "")
                        if isinstance(content, str) and content:
                            streamed_parts.append(content)

            else:
                for message, metadata in graph.stream(
                    graph_input,
                    config=config,
                    stream_mode="messages",
                ):
                    if metadata.get("langgraph_node") == "llm":
                        content = content_to_text(
                            getattr(message, "content", "")
                        )

                        if content:
                            streamed_parts.append(content)

            # Get the final graph state.
            state = graph.get_state(config)
            messages = state.values.get("messages", [])

            # Use the final AI message as the source of truth.
            last_message = next(
                (
                    message
                    for message in reversed(messages)
                    if getattr(message, "type", None) == "ai"
                ),
                None,
            )

            answer = (
                content_to_text(last_message.content)
                if last_message else ""
            )

            # Fallback only if the graph has no final AI content.
            if not answer:
                answer = "".join(streamed_parts)

            if not answer:
                raise RuntimeError("The graph returned an empty answer.")

            # Send the answer exactly once.
            yield make_event("token", content=answer)

            additional = (
                last_message.additional_kwargs
                if last_message else {}
            )

            print("\n========== API CHUNKS CHECK ==========")
            print(
                "Retrieved chunks in final message:",
                len(additional.get("retrieved_chunks", []))
            )
            print("Additional kwargs keys:", list(additional.keys()))
            print("======================================")

            print("\n========== STREAMING API CHECK ==========")
            print("Chunks:", len(additional.get("retrieved_chunks", [])))
            print("Message ID:", last_message.id if last_message else None)
            print("=========================================")


            if last_message:
                print(
                    "FINAL MESSAGE USAGE:",
                    last_message.usage_metadata
                )

            # Calculate API usage.
            usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

            for model_usage in callback.usage_metadata.values():
                if not isinstance(model_usage, dict):
                    continue

                input_tokens = model_usage.get("input_tokens", 0)
                output_tokens = model_usage.get("output_tokens", 0)

                usage["input_tokens"] += input_tokens
                usage["output_tokens"] += output_tokens
                usage["total_tokens"] += model_usage.get(
                    "total_tokens",
                    input_tokens + output_tokens,
                )

            # if last_message and last_message.type == "ai":
            # Get the context usage calculated inside rag_graph.py.
            graph_context_usage = (
                state.values.get("context_usage") or {}
            )

            # Use the answer-generation prompt size for context utilization.
            prompt_tokens = graph_context_usage.get("prompt_tokens", 0)

            # General-chat fallback when graph context metrics are unavailable.
            if not prompt_tokens:
                prompt_tokens = usage["input_tokens"]

            context_usage = {
                **graph_context_usage,
                "input_tokens": prompt_tokens,
                "prompt_tokens": prompt_tokens,
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
            }

            # Persist both usage metrics with the final AI message.
            if last_message and last_message.type == "ai":
                last_message.additional_kwargs["usage"] = usage
                last_message.additional_kwargs["context_usage"] = context_usage

                graph.update_state(
                    config,
                    {"messages": [last_message]}
                )

            yield make_event(
                "done",
                response_id=last_message.id if last_message else None,
                answer_source=additional.get(
                    "answer_source",
                    "LLM knowledge",
                ),
                feedback_query=additional.get(
                    "feedback_query",
                    query
                ),
                feedback_context=additional.get(
                    "feedback_context",
                    []
                ),
                retrieved_chunks=additional.get("retrieved_chunks", []),
                usage=usage,
                context_usage=context_usage,
                
            )

        except Exception as exc:
            yield make_event("error", message=str(exc))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )





# ---------- Feedback ----------

class FeedbackRequest(BaseModel):
    chat_id: str
    response_id: str
    query: str
    response: str
    feedback: str
    context: object = None
    answer_source: str = ""
    project_id: str | None = None


@app.post("/api/feedback")
def submit_feedback(body: FeedbackRequest):
    chats = load_chats()
    chat = chats.get(body.chat_id)

    if not chat:
        raise HTTPException(404, "Chat not found.")

    if body.feedback not in ("good", "bad"):
        raise HTTPException(400, "Feedback must be good or bad.")

    project_id = body.project_id or chat.get("rag_id")

    catalog = load_catalog()
    project_name = (
        catalog.get(project_id, {}).get("name")
        if project_id else None
    )

    save_feedback(
        chat_id=body.chat_id,
        mode=chat.get("mode", "general"),
        chat_name=chat.get("title", "New Chat"),
        response_id=body.response_id,
        query=body.query,
        response=body.response,
        feedback=body.feedback,
        context=body.context,
        answer_source=body.answer_source,
        project_id=project_id,
        project_name=project_name,
    )

    return {"success": True}



def get_project_document_path(rag_id: str, filename: str):
    catalog = load_catalog()
    project = catalog.get(rag_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    # Only allow files registered in this project's catalog.
    if filename not in project.get("files", []):
        raise HTTPException(status_code=404, detail="Document not found.")

    # Prevent path traversal and access outside the project directory.
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    project_dir = (RAGS_DIR / rag_id).resolve()
    file_path = (project_dir / filename).resolve()

    if file_path.parent != project_dir or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")

    if Path(rag_id).name != rag_id:
        raise HTTPException(status_code=404, detail="Project not found.")


    return file_path




@app.get("/api/projects/{rag_id}/documents")
async def get_project_documents(rag_id: str):
    catalog = load_catalog()
    project = catalog.get(rag_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    documents = []

    for filename in project.get("files", []):
        try:
            path = get_project_document_path(rag_id, filename)
        except HTTPException:
            continue

        documents.append({
            "name": filename,
            "extension": path.suffix.lower()
        })

    return {"documents": documents}


@app.get("/api/projects/{rag_id}/documents/{filename}/preview")
async def preview_project_document(rag_id: str, filename: str):
    path = get_project_document_path(rag_id, filename)
    extension = path.suffix.lower()

    if extension == ".pdf":
        return Response(
            content=path.read_bytes(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"'
            }
        )

    if extension == ".txt":
        return {
            "type": "txt",
            "text": path.read_text(encoding="utf-8", errors="replace")
        }

    if extension == ".docx":
        from docx import Document as WordDocument

        document = WordDocument(str(path))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        tables = [
            [
                [cell.text for cell in row.cells]
                for row in table.rows
            ]
            for table in document.tables
        ]

        return {
            "type": "docx",
            "paragraphs": paragraphs,
            "tables": tables
        }

    if extension == ".csv":
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline=""
        ) as file:
            rows = list(csv.reader(file))

        return {
            "type": "table",
            "columns": rows[0] if rows else [],
            "rows": rows[1:201] if rows else [],
            "truncated": len(rows) > 201
        }

    if extension == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(
            filename=path,
            read_only=True,
            data_only=True
        )

        sheets = []

        try:
            for worksheet in workbook.worksheets:
                rows = worksheet.iter_rows(
                    min_row=1,
                    max_row=201,
                    values_only=True
                )

                values = list(rows)

                def serialize_cell(value):
                    if isinstance(value, (datetime, date)):
                        return value.isoformat()
                    return value

                values = [
                    [serialize_cell(cell) for cell in row]
                    for row in values
                ]

                sheets.append({
                    "name": worksheet.title,
                    "columns": values[0] if values else [],
                    "rows": values[1:] if values else [],
                    "truncated": worksheet.max_row > 201
                })
        finally:
            workbook.close()

        return {
            "type": "xlsx",
            "sheets": sheets
        }

    raise HTTPException(
        status_code=415,
        detail="This document format is not supported."
    )


@app.get("/api/projects/{rag_id}/documents/{filename}/download")
async def download_project_document(rag_id: str, filename: str):
    from fastapi.responses import FileResponse

    path = get_project_document_path(rag_id, filename)

    return FileResponse(
        path,
        filename=filename,
        media_type="application/octet-stream"
    )







@app.get("/health")
def health():
    return {"status": "ok"}
