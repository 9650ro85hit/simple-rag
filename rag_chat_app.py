import json
import shutil
import sqlite3
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.callbacks import UsageMetadataCallbackHandler
from rag_graph import build_rag_graph
from feedback_storage import save_feedback, get_feedback

sys.path.insert(0, str(Path.cwd().parent))
from RAG.rag.llm_call import llm
from rag_storage import create_rag, load_catalog, load_vector_store, BASE_DIR, RAGS_DIR
from rag_graph import build_rag_graph

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")
CHATS_FILE = BASE_DIR / "chats.json"
CATALOG_FILE = BASE_DIR / "rags.json"
DB_FILE = BASE_DIR / "chat_memory.sqlite"


# ---------- JSON storage ----------
def _load_dict(path, label):
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}; restore a valid backup.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return data


def load_chats():
    return _load_dict(CHATS_FILE, "chats.json")


def _save_dict(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        # Pass the object as the first argument. This fixes dump() missing 'obj'.
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
    temp.replace(path)


def save_chats(chats):
    _save_dict(CHATS_FILE, chats)

def extract_token_usage(callback):
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    found_usage = False

    for model_usage in callback.usage_metadata.values():
        if not isinstance(model_usage, dict):
            continue

        found_usage = True

        input_tokens = model_usage.get("input_tokens", 0)
        output_tokens = model_usage.get("output_tokens", 0)

        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["total_tokens"] += model_usage.get(
            "total_tokens",
            input_tokens + output_tokens
        )

    return totals if found_usage else None


def save_catalog_file(catalog):
    _save_dict(CATALOG_FILE, catalog)


def rename_chat(chat_id, name):
    name = name.strip()
    if not name:
        raise ValueError("Chat name cannot be empty.")
    chats = load_chats()
    if chat_id not in chats:
        raise ValueError("Chat no longer exists.")
    chats[chat_id]["title"] = name
    save_chats(chats)


def rename_project(rag_id, name):
    name = name.strip()
    if not name:
        raise ValueError("Project name cannot be empty.")
    catalog = load_catalog()
    if rag_id not in catalog:
        raise ValueError("Project no longer exists.")
    catalog[rag_id]["name"] = name
    save_catalog_file(catalog)


# ---------- SQLite checkpoints ----------
@st.cache_resource
def get_checkpointer():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def delete_thread_checkpoints(thread_id):
    saver = get_checkpointer()
    conn = saver.conn
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    with conn:
        for table in ("writes", "checkpoints", "checkpoint_blobs", "blobs"):
            if table not in tables:
                continue
            columns = {
                row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "thread_id" in columns:
                conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))


# ---------- Graphs ----------
@st.cache_resource
def get_graph(rag_id):
    return build_rag_graph(
        vector_store=load_vector_store(rag_id),
        checkpointer=get_checkpointer(),
    )



def call_normal_llm(state):
    messages = state["messages"]

    # The latest message is the current user query.
    query = messages[-1].content if messages else ""

    # Preserve the conversation history actually passed to the LLM.
    history = [
        {
            "role": message.type,
            "content": message.content
        }
        for message in messages[:-1]
    ]

    # response = llm.invoke(messages)

    response = llm.invoke([
SystemMessage(content="""
You are a helpful AI assistant.

Format every response clearly and consistently using standard Markdown.

Follow these guidelines:
- Use appropriate headings, paragraphs, lists, bold text, tables, and code blocks whenever useful.
- Format all mathematical expressions using valid LaTeX delimiters: \\( ... \\) for inline mathematics and \\[ ... \\] for display mathematics.
- Never leave mathematical expressions as unformatted plain text.
- Use properly formatted code blocks with the appropriate language identifier.
- Keep the response readable, well-structured, and relevant to the user's question.
- Avoid unnecessary formatting and repetition.
- Adapt the response structure to the type of question.
- Keep answers concise unless more detail is needed.
""")
   ,
        *messages
    ])



    return {
        "messages": [
            AIMessage(
                content=response.content,
                additional_kwargs={
                    "answer_source": "LLM knowledge",
                    "feedback_query": query,
                    "feedback_context": history
                },
                response_metadata=response.response_metadata,
                usage_metadata=response.usage_metadata
            )
        ]
    }



@st.cache_resource
def get_general_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_normal_llm)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    return builder.compile(checkpointer=get_checkpointer())


# ---------- Create / delete ----------
def create_chat(mode, rag_id=None):
    thread_id = str(uuid4())
    chats = load_chats()
    chats[thread_id] = {"title": "New Chat", "mode": mode, "rag_id": rag_id}
    save_chats(chats)
    st.session_state.current_thread_id = thread_id
    return thread_id


def delete_chat(chat_id):
    chats = load_chats()
    if chat_id not in chats:
        return
    # Only remove metadata after checkpoint deletion succeeds.
    delete_thread_checkpoints(chat_id)
    chats.pop(chat_id, None)
    save_chats(chats)
    if st.session_state.get("current_thread_id") == chat_id:
        st.session_state.current_thread_id = None


def delete_project(rag_id):
    catalog = load_catalog()
    if rag_id not in catalog:
        return
    chats = load_chats()
    ids = [
        cid for cid, chat in chats.items()
        if chat.get("rag_id") == rag_id and chat.get("mode", "project") == "project"
    ]
    for cid in ids:
        delete_thread_checkpoints(cid)
    for cid in ids:
        chats.pop(cid, None)
    save_chats(chats)

    project_dir = (RAGS_DIR / rag_id).resolve()
    if project_dir.parent != RAGS_DIR.resolve():
        raise ValueError("Invalid project storage path.")
    get_graph.clear()
    if project_dir.exists():
        shutil.rmtree(project_dir)
    catalog.pop(rag_id, None)
    save_catalog_file(catalog)
    if st.session_state.get("selected_rag_id") == rag_id:
        st.session_state.selected_rag_id = None
    if st.session_state.get("current_thread_id") in ids:
        st.session_state.current_thread_id = None


# ---------- Dialogs ----------
@st.dialog("Create a RAG project")
def create_rag_dialog():
    name = st.text_input("Project name", placeholder="e.g. HR Documents")
    files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt", "docx", "csv", "xlsx"], accept_multiple_files=True)
    if st.button("Create project", type="primary", use_container_width=True):
        if not name.strip():
            st.warning("Enter a project name.")
            return
        if not files:
            st.warning("Upload at least one document.")
            return
        try:
            with st.spinner("Processing documents and creating embeddings..."):
                rag_id = create_rag(name.strip(), files)
            st.session_state.selected_rag_id = rag_id
            st.session_state.current_thread_id = None
            st.session_state.chat_type_selector = "Project Chat"
            st.rerun()
        except Exception as exc:
            st.error(f"Could not create project: {exc}")


@st.dialog("Rename chat")
def rename_chat_dialog(chat_id):
    chat = load_chats().get(chat_id)
    if not chat:
        st.error("This chat no longer exists.")
        return
    name = st.text_input("Chat name", value=chat.get("title", "New Chat"))
    a, b = st.columns(2)
    with a:
        if st.button("Save name", type="primary", use_container_width=True):
            try:
                rename_chat(chat_id, name)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not rename chat: {exc}")
    with b:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("Rename project")
def rename_project_dialog(rag_id):
    project = load_catalog().get(rag_id)
    if not project:
        st.error("This project no longer exists.")
        return
    name = st.text_input("Project name", value=project.get("name", "Untitled project"))
    a, b = st.columns(2)
    with a:
        if st.button("Save name", type="primary", use_container_width=True):
            try:
                rename_project(rag_id, name)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not rename project: {exc}")
    with b:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("Delete chat?")
def confirm_delete_chat_dialog(chat_id):
    chat = load_chats().get(chat_id)
    if not chat:
        st.info("This chat has already been deleted.")
        return
    st.write(f'Delete **{chat.get("title", "New Chat")}** and its saved history?')
    st.caption("This cannot be undone.")
    a, b = st.columns(2)
    with a:
        if st.button("Delete chat", type="primary", use_container_width=True):
            try:
                delete_chat(chat_id)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete chat: {exc}")
    with b:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("Delete project?")
def confirm_delete_project_dialog(rag_id):
    project = load_catalog().get(rag_id)
    if not project:
        st.info("This project has already been deleted.")
        return
    st.write(
        f'Delete **{project.get("name", "Untitled project")}**, all associated chats, '
        "uploaded files, and the FAISS index?"
    )
    st.warning("This cannot be undone.")
    a, b = st.columns(2)
    with a:
        if st.button("Delete project", type="primary", use_container_width=True):
            try:
                delete_project(rag_id)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete project: {exc}")
    with b:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ---------- Session state ----------
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "selected_rag_id" not in st.session_state:
    st.session_state.selected_rag_id = None
if "chat_type_selector" not in st.session_state:
    st.session_state.chat_type_selector = "Project Chat"


# ---------- UI theme ----------
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Dark"

dark_mode = st.session_state.ui_theme == "Dark"

if dark_mode:
    bg = "#0F172A"
    sidebar_bg = "#172438"
    panel_bg = "#22334A"
    text_color = "#F1F5F9"
    muted_color = "#A8B6C8"
    border_color = "#34465F"
    accent_color = "#60A5FA"
else:
    bg = "#F8FAFC"
    sidebar_bg = "#FFFFFF"
    panel_bg = "#F1F5F9"
    text_color = "#172033"
    muted_color = "#64748B"
    border_color = "#D8E1EC"
    accent_color = "#2563EB"


st.markdown(
    f"""
    <style>

    /* Main application */
    [data-testid="stAppViewContainer"] {{
        background: {bg};
        color: {text_color};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {sidebar_bg};
        border-right: 1px solid {border_color};
    }}

    [data-testid="stSidebar"] * {{
        color: {text_color};
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: {text_color} !important;
    }}

    /* Project expanders */
    [data-testid="stSidebar"] [data-testid="stExpander"] {{
        background: {sidebar_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
        background: {panel_bg} !important;
        color: {text_color} !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
        opacity: 1 !important;
    }}

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {{
        background: {panel_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span {{
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebar"] .stButton > button:hover {{
        background: {accent_color} !important;
        border-color: {accent_color} !important;
    }}

    [data-testid="stSidebar"] .stButton > button:hover p,
    [data-testid="stSidebar"] .stButton > button:hover span {{
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }}

    /* All popovers - including project/chat options */
    [data-testid="stPopoverBody"] {{
        background: {panel_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
        padding: 10px !important;
        min-width: 150px !important;
        max-width: 190px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25) !important;
    }}

    [data-testid="stPopoverBody"] * {{
        color: {text_color} !important;
    }}

    [data-testid="stPopoverBody"] .stButton > button {{
        width: 100% !important;
        background: {sidebar_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        min-height: 34px !important;
        font-size: 13px !important;
    }}

    [data-testid="stPopoverBody"] .stButton > button:hover {{
        background: {accent_color} !important;
        color: #FFFFFF !important;
        border-color: {accent_color} !important;
    }}

    /* Popover trigger buttons */
    [data-testid="stPopover"] > button {{
        background: {panel_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 9px !important;
    }}

    [data-testid="stPopover"] > button svg {{
        color: {text_color} !important;
        fill: {text_color} !important;
    }}

    /* Header and theme toggle */
    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* General buttons */
    .stButton > button {{
        color: {text_color};
        border-color: {border_color};
    }}

    /* Chat input */
    [data-testid="stChatInput"] {{
        background: {sidebar_bg};
        border: 1px solid {border_color};
        border-radius: 14px;
    }}

    [data-testid="stChatInput"] textarea {{
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: {muted_color} !important;
        -webkit-text-fill-color: {muted_color} !important;
    }}





    /* Chat conversation - both themes */
    [data-testid="stChatMessage"] {{
        background-color: {panel_bg} !important;
        color: {text_color} !important;
    }}

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {{
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
    }}

    
    
    [data-testid="stChatMessage"] button {{
        min-height: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        width: 28px !important;
        padding: 0 !important;
        font-size: 12px !important;
        border-radius: 6px !important;
    }}

    
    [data-testid="stChatMessage"] button p {{
        margin: 0 !important;
        padding: 0 !important;
    }}


    </style>
    """,
    unsafe_allow_html=True
)




# ---------- Sidebar: project/chat rows and three-dot menus ----------


# ---------- AI Workbench Sidebar ----------

catalog = load_catalog()
chats = load_chats()

with st.sidebar:

    # Branding
    st.markdown("## ✦ AI WORKBENCH")
    st.caption("Your personal AI workspace")

    st.divider()

    # Create project
    if st.button(
        "＋  New project",
        key="workbench_new_project",
        use_container_width=True,
        type="primary"
    ):
        create_rag_dialog()

    st.markdown("#### YOUR PROJECTS")

    if not catalog:
        st.caption("No projects yet.")

    # Project list
    for rid, project in catalog.items():

        project_name = project.get("name", "Untitled project")

        is_selected = (
            st.session_state.get("selected_rag_id") == rid
        )

        with st.expander(
            f"📁  {project_name}",
            expanded=is_selected
        ):

            # Project actions
            col1, col2 = st.columns([4, 1])

            with col1:
                if st.button(
                    "Open project",
                    key=f"open_project_{rid}",
                    use_container_width=True
                ):
                    st.session_state.selected_rag_id = rid
                    st.session_state.current_thread_id = None
                    st.session_state.chat_type_selector = "Project Chat"
                    st.rerun()

            with col2:
                with st.popover("⋯", help="Project options"):
                    if st.button(
                        "Rename",
                        key=f"rename_project_{rid}"
                    ):
                        rename_project_dialog(rid)

                    if st.button(
                        "Delete",
                        key=f"delete_project_{rid}"
                    ):
                        confirm_delete_project_dialog(rid)

            # New chat in this project
            if st.button(
                "＋ New chat",
                key=f"new_project_chat_{rid}",
                use_container_width=True
            ):
                st.session_state.selected_rag_id = rid
                st.session_state.chat_type_selector = "Project Chat"
                create_chat("project", rid)
                st.rerun()

            # Project chat history
            project_chats = {
                cid: item
                for cid, item in chats.items()
                if item.get("rag_id") == rid
                and item.get("mode", "project") == "project"
            }

            project_chat_ids = list(reversed(list(project_chats)))

            for cid in project_chat_ids:

                item = project_chats[cid]

                active = (
                    cid == st.session_state.get("current_thread_id")
                )

                col1, col2 = st.columns([5, 1])

                with col1:
                    if st.button(
                        f"{'● ' if active else '○ '}{item.get('title', 'New Chat')}",
                        key=f"project_chat_{cid}",
                        use_container_width=True,
                        type="primary" if active else "secondary"
                    ):
                        st.session_state.current_thread_id = cid
                        st.session_state.selected_rag_id = rid
                        st.session_state.chat_type_selector = "Project Chat"
                        st.rerun()

                with col2:
                    with st.popover("⋯", help="Chat options"):
                        if st.button(
                            "Rename",
                            key=f"rename_project_chat_{cid}"
                        ):
                            rename_chat_dialog(cid)

                        if st.button(
                            "Delete",
                            key=f"delete_project_chat_{cid}"
                        ):
                            confirm_delete_chat_dialog(cid)

    st.divider()

    # General chats
    st.markdown("#### GENERAL CHATS")

    if st.button(
        "＋ New general chat",
        key="new_general_chat",
        use_container_width=True
    ):
        st.session_state.chat_type_selector = "General Chat"
        create_chat("general")
        st.rerun()

    general_chats = {
        cid: item
        for cid, item in chats.items()
        if item.get("mode") == "general"
    }

    general_chat_ids = list(reversed(list(general_chats)))

    if not general_chat_ids:
        st.caption("No general chats yet.")

    for cid in general_chat_ids:

        item = general_chats[cid]

        active = (
            cid == st.session_state.get("current_thread_id")
        )

        col1, col2 = st.columns([5, 1])

        with col1:
            if st.button(
                f"{'● ' if active else '○ '}{item.get('title', 'New Chat')}",
                key=f"general_chat_{cid}",
                use_container_width=True,
                type="primary" if active else "secondary"
            ):
                st.session_state.current_thread_id = cid
                st.session_state.chat_type_selector = "General Chat"
                st.rerun()

        with col2:
            with st.popover("⋯", help="Chat options"):
                if st.button(
                    "Rename",
                    key=f"rename_general_chat_{cid}"
                ):
                    rename_chat_dialog(cid)

                if st.button(
                    "Delete",
                    key=f"delete_general_chat_{cid}"
                ):
                    confirm_delete_chat_dialog(cid)

# ---------- Resolve active conversation ----------
catalog = load_catalog()
chats = load_chats()
chat_type = st.session_state.chat_type_selector
thread_id = st.session_state.get("current_thread_id")
chat = chats.get(thread_id) if thread_id else None

if chat_type == "Project Chat":
    if not catalog:
        st.title("RAG Document Chat")
        st.info("Create a project or switch to General Chat.")
        st.stop()
    rag_id = st.session_state.get("selected_rag_id")
    if rag_id not in catalog:
        rag_id = next(iter(catalog))
        st.session_state.selected_rag_id = rag_id
    if not chat or chat.get("mode", "project") != "project" or chat.get("rag_id") != rag_id:
        project_ids = [
            cid for cid, item in chats.items()
            if item.get("rag_id") == rag_id and item.get("mode", "project") == "project"
        ]
        if project_ids:
            thread_id = project_ids[-1]
            st.session_state.current_thread_id = thread_id
        else:
            thread_id = create_chat("project", rag_id)
        chats = load_chats()
        chat = chats[thread_id]
else:
    if not chat or chat.get("mode") != "general":
        general_ids = [cid for cid, item in chats.items() if item.get("mode") == "general"]
        if general_ids:
            thread_id = general_ids[-1]
            st.session_state.current_thread_id = thread_id
        else:
            thread_id = create_chat("general")
        chats = load_chats()
        chat = chats[thread_id]




# ---------- Active workspace ----------

mode = chat.get("mode", "project")
rag_id = chat.get("rag_id")

if mode == "project":

    if rag_id not in catalog:
        st.error("The project associated with this chat no longer exists.")
        st.stop()

    graph = get_graph(rag_id)

    workspace_name = catalog[rag_id].get(
        "name",
        "Untitled project"
    )

    workspace_description = "Project workspace"

else:

    graph = get_general_graph()

    workspace_name = "General Chat"
    workspace_description = "Normal LLM · No document retrieval"


# ---------- Main header ----------

# ---------- Main header and token tracker ----------

saved_chat = load_chats().get(thread_id, {})

token_usage = saved_chat.get("token_usage", {})
last_usage = saved_chat.get("last_token_usage", {})

header_left, token_col, theme_col = st.columns(
    [7, 0.8, 1.5],
    vertical_alignment="center"
)

with header_left:
    st.markdown(f"### {workspace_name}")
    st.caption(workspace_description)

# with token_col:

#     with st.popover("◉", help="Conversation token usage"):

#         st.markdown("**Token usage**")
#         st.caption("Actual usage reported by the model")

#         st.metric(
#             "Total tokens",
#             f"{token_usage.get('total_tokens', 0):,}"
#         )

#         st.divider()

#         st.caption("Conversation totals")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.metric(
#                 "Input",
#                 f"{token_usage.get('input_tokens', 0):,}"
#             )

#         with col2:
#             st.metric(
#                 "Output",
#                 f"{token_usage.get('output_tokens', 0):,}"
#             )

#         st.divider()

#         st.caption("Last request")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.metric(
#                 "Input",
#                 f"{last_usage.get('input_tokens', 0):,}"
#             )

#         with col2:
#             st.metric(
#                 "Output",
#                 f"{last_usage.get('output_tokens', 0):,}"
#             )

#         if not token_usage:
#             st.caption(
#                 "Usage will appear after a supported model response."
#             )

with theme_col:

    theme_label = "🌙 Dark"

    if st.button(
        theme_label,
        key="toggle_theme",
        use_container_width=True
    ):
        st.session_state.ui_theme = (
             "Dark"
        )
        st.rerun()

st.divider()

config = {"configurable": {"thread_id": thread_id}}

# ---------- Display history ----------
snapshot = graph.get_state(config)
messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []

context_usage = (
    snapshot.values.get("context_usage", {})
    if snapshot and snapshot.values
    else {}
)

# General chat does not use the RAG graph's context_usage field.
if mode == "general":
    total_tokens = count_tokens_approximately(messages)

    context_usage = {
        "prompt_tokens": total_tokens,
        "history_tokens": total_tokens,
        "summary_tokens": 0,
        "recent_history_tokens": total_tokens,
        "retrieved_context_tokens": 0,
        "query_tokens": 0,
        "summary_threshold": 6000,
        "summarization_triggered": False
    }



for message in messages:

    if isinstance(message, HumanMessage):
        role = "user"

    elif isinstance(message, AIMessage):
        role = "assistant"

    else:
        continue

    with st.chat_message(role):

        if isinstance(message.content, str):
            st.markdown(message.content)
        else:
            st.write(message.content)


        # Display answer source and feedback controls
        if isinstance(message, AIMessage):

            metadata = message.additional_kwargs or {}

            source = metadata.get("answer_source")

            if source == "Uploaded documents":
                st.caption("📄 Source: Uploaded documents")

            elif source == "Online search":
                st.caption("🌐 Source: Online search")

            elif source == "LLM knowledge":
                st.caption("🤖 Source: LLM knowledge")

            # Only show feedback for responses with captured metadata.
            feedback_query = metadata.get("feedback_query")

            if feedback_query and message.id:

                response_id = message.id

                existing_feedback = get_feedback(response_id)

                # Keep the feedback controls at the bottom-right.
                spacer, up_col, down_col = st.columns(
                    [9, 0.35, 0.35],
                    vertical_alignment="center"
                )

                with up_col:
                    if st.button(
                        "👍",
                        key=f"feedback_good_{response_id}",
                        help="Good answer",
                        disabled=existing_feedback is not None,
                        type="secondary"
                    ):
                        save_feedback(
                            chat_id=thread_id,
                            mode=mode,
                            chat_name=chat.get("title", "New Chat"),
                            response_id=response_id,
                            query=feedback_query,
                            response=message.content,
                            feedback="good",
                            context=metadata.get("feedback_context", []),
                            answer_source=source,
                            project_id=rag_id if mode == "project" else None,
                            project_name=(
                                catalog.get(rag_id, {}).get("name")
                                if mode == "project"
                                else None
                            )
                        )
                        st.rerun()

                with down_col:
                    if st.button(
                        "👎",
                        key=f"feedback_bad_{response_id}",
                        help="Bad answer",
                        disabled=existing_feedback is not None,
                        type="secondary"
                    ):
                        save_feedback(
                            chat_id=thread_id,
                            mode=mode,
                            chat_name=chat.get("title", "New Chat"),
                            response_id=response_id,
                            query=feedback_query,
                            response=message.content,
                            feedback="bad",
                            context=metadata.get("feedback_context", []),
                            answer_source=source,
                            project_id=rag_id if mode == "project" else None,
                            project_name=(
                                catalog.get(rag_id, {}).get("name")
                                if mode == "project"
                                else None
                            )
                        )
                        st.rerun()

                if existing_feedback == "good":
                    st.caption("You marked this answer as helpful.")

                elif existing_feedback == "bad":
                    st.caption("You marked this answer as not helpful.")




# ---------- Empty conversation welcome screen ----------

if not messages:

    if mode == "project":

        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 90px 20px 55px 20px;
            ">
                <h2>What are we exploring today?</h2>
                <p style="opacity: 0.65; font-size: 15px;">
                    Ask questions about your documents and research.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 90px 20px 55px 20px;
            ">
                <h2>What would you like to work on?</h2>
                <p style="opacity: 0.65; font-size: 15px;">
                    Start a conversation with your AI assistant.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------- New message ----------
# ---------- New message and token usage ----------

threshold = context_usage.get("summary_threshold", 6000)
history_tokens = context_usage.get("history_tokens", 0)

usage_percent = min(
    100,
    round((history_tokens / threshold) * 100) if threshold else 0
)

input_col, token_col = st.columns(
    [12, 1],
    vertical_alignment="bottom"
)

with input_col:
    query = st.chat_input(
        "Ask about your documents..."
        if mode == "project"
        else "Ask me anything..."
    )

with token_col:
    with st.popover(
        f"◉ {usage_percent}%",
        help="View conversation token usage"
    ):

        st.markdown("### Context usage")

        # Circular usage indicator
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; padding:12px;">
                <div style="
                    width:100px;
                    height:100px;
                    border-radius:50%;
                    background:conic-gradient(
                        #4F8BF9 {usage_percent}%,
                        #D9DDE5 {usage_percent}%
                    );
                    display:flex;
                    align-items:center;
                    justify-content:center;
                ">
                    <div style="
                        width:76px;
                        height:76px;
                        border-radius:50%;
                        background:var(--background-color);
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:20px;
                        font-weight:600;
                    ">
                        {usage_percent}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            f"Conversation history / {threshold:,}-token summarization threshold"
        )

        st.divider()

        st.metric(
            "Total prompt tokens (estimated)",
            f'{context_usage.get("prompt_tokens", 0):,}'
        )

        st.metric(
            "Conversation history tokens",
            f'{context_usage.get("history_tokens", 0):,}'
        )

        st.metric(
            "Summary tokens",
            f'{context_usage.get("summary_tokens", 0):,}'
        )

        st.metric(
            "Recent conversation tokens",
            f'{context_usage.get("recent_history_tokens", 0):,}'
        )

        st.metric(
            "Retrieved context tokens",
            f'{context_usage.get("retrieved_context_tokens", 0):,}'
        )

        st.metric(
            "Current query tokens",
            f'{context_usage.get("query_tokens", 0):,}'
        )

        if context_usage.get("summarization_triggered", False):
            st.success("Conversation summarization triggered.")
        else:
            st.info("Conversation summarization not triggered.")

        st.caption(
            "Token counts are approximate. The history percentage tracks "
            "the summarization threshold, not the model's context-window limit."
        )



if query:
    try:
        if chat.get("title") == "New Chat":
            chat["title"] = " ".join(query.split()[:6])
            save_chats(chats)

        # Capture token usage
        usage_callback = UsageMetadataCallbackHandler()

        run_config = {
            **config,
            "callbacks": [usage_callback]
        }

        # Prepare graph input
        if mode == "project":
            graph_input = {
                "query": query,
                "messages": [
                    HumanMessage(content=query)
                ]
            }
        else:
            graph_input = {
                "messages": [
                    HumanMessage(content=query)
                ]
            }

        # Display user question
        with st.chat_message("user"):
            st.markdown(query)

        # Display streamed assistant answer
        with st.chat_message("assistant"):

            response_placeholder = st.empty()
            response_placeholder.markdown("_Thinking..._")

            full_response = ""

            for event in graph.stream(
                graph_input,
                config=run_config,
                stream_mode="custom"
            ):

                if (
                    isinstance(event, dict)
                    and event.get("type") == "answer_token"
                ):

                    token = event.get("content", "")

                    full_response += token

                    response_placeholder.markdown(full_response)

        # Extract actual token usage
        current_usage = extract_token_usage(usage_callback)

        if current_usage:

            updated_chats = load_chats()
            current_chat = updated_chats.get(thread_id)

            if current_chat:

                totals = current_chat.setdefault(
                    "token_usage",
                    {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    }
                )

                for key in totals:
                    totals[key] = (
                        totals.get(key, 0) + current_usage[key]
                    )

                current_chat["last_token_usage"] = current_usage

                save_chats(updated_chats)

        st.rerun()

    except Exception as exc:
        st.error(f"Something went wrong: {exc}")




