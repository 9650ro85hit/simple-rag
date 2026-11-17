
import json
import re
from datetime import datetime, timezone
from pathlib import Path


# Permanent feedback storage directory
FEEDBACK_DIR = Path(__file__).resolve().parent / "rag_feedback"
INDEX_FILE = FEEDBACK_DIR / "feedback_files.json"


def _safe_filename(name):
    """Convert a project/chat name into a safe filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", str(name))
    name = re.sub(r"\s+", "_", name.strip())
    return name[:70] or "untitled"


def _load_feedback_index():
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    if not INDEX_FILE.exists():
        return {}

    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        raise RuntimeError(
            "Feedback index is unreadable. Restore a valid backup before continuing."
        )


def _save_feedback_index(index):
    temp_file = INDEX_FILE.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    temp_file.replace(INDEX_FILE)


def _get_feedback_file(
    chat_id,
    mode,
    chat_name,
    project_id=None,
    project_name=None
):
    index = _load_feedback_index()

    if mode == "project":
        key = f"project:{project_id}"
        filename = (
            f"{_safe_filename(project_name)}_project_chat_"
            f"{_safe_filename(project_id)[:8]}.jsonl"
        )
    else:
        key = f"general:{chat_id}"
        filename = (
            f"{_safe_filename(chat_name)}_general_chat_"
            f"{_safe_filename(chat_id)[:8]}.jsonl"
        )

    # Once assigned, keep the same file even if the project/chat is renamed.
    if key not in index:
        index[key] = filename
        _save_feedback_index(index)

    return FEEDBACK_DIR / index[key]


def save_feedback(
    chat_id,
    mode,
    chat_name,
    response_id,
    query,
    response,
    feedback,
    context,
    answer_source,
    project_id=None,
    project_name=None
):
    if feedback not in ("good", "bad"):
        raise ValueError("Feedback must be 'good' or 'bad'.")

    if not response_id:
        raise ValueError("Response ID is missing.")

    file_path = _get_feedback_file(
        chat_id=chat_id,
        mode=mode,
        chat_name=chat_name,
        project_id=project_id,
        project_name=project_name
    )

    # Prevent duplicate feedback for the same response.
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("response_id") == response_id:
                    return False

    record = {
        "response_id": response_id,
        "chat_id": chat_id,
        "chat_name": chat_name,
        "project_id": project_id,
        "project_name": project_name,
        "mode": mode,
        "query": query,
        "response": response,
        "feedback": feedback,
        "answer_source": answer_source,
        "context": context or [],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    with file_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return True


def get_feedback(response_id):
    """Find previously submitted feedback for a response."""
    if not FEEDBACK_DIR.exists():
        return None

    for file_path in FEEDBACK_DIR.glob("*.jsonl"):
        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)

                if record.get("response_id") == response_id:
                    return record.get("feedback")

    return None
