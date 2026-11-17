
# import json
# from pathlib import Path
# from uuid import uuid4
import re
import json
import os
import shutil

from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4


import streamlit as st
import fitz
import pytesseract

from PIL import Image

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import csv

from docx import Document as WordDocument
from openpyxl import load_workbook

from collections import Counter
# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------


# Windows and Docker/Linux compatible Tesseract configuration

TESSERACT_CMD = (
    os.getenv("TESSERACT_CMD")
    or shutil.which("tesseract")
    or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

BASE_DIR = Path(__file__).resolve().parent
RAGS_DIR = BASE_DIR / "rags"
CATALOG_FILE = BASE_DIR / "rags.json"

RAGS_DIR.mkdir(parents=True, exist_ok=True)


# Load embedding model once per Python process
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# UI ADAPTER
# --------------------------------------------------

class NoOpProgress:

    def progress(self, *args, **kwargs):
        pass

    def empty(self):
        pass


class NoOpStatus:

    def update(self, *args, **kwargs):
        pass


class NoOpUI:

    def status(self, *args, **kwargs):
        return NoOpStatus()

    def progress(self, *args, **kwargs):
        return NoOpProgress()

    def spinner(self, *args, **kwargs):
        return nullcontext()

    def success(self, *args, **kwargs):
        pass


def get_ui(show_ui=True):

    if show_ui:
        return st

    return NoOpUI()

# --------------------------------------------------
# 2. CATALOG MANAGEMENT
# --------------------------------------------------

def load_catalog():

    if not CATALOG_FILE.exists():
        return {}

    if CATALOG_FILE.stat().st_size == 0:
        return {}

    try:
        with open(CATALOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"{CATALOG_FILE} must contain a JSON object."
            )

        return data

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {CATALOG_FILE}."
        ) from e


def save_catalog(catalog):

    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)


# --------------------------------------------------
# 3. DOCUMENT EXTRACTION
# --------------------------------------------------

# def extract_pdf_documents(file_path, filename):

#     documents = []

#     pdf = fitz.open(str(file_path))
#     total_pages = len(pdf)

#     progress = st.progress(
#         0,
#         text=f"Extracting PDF: 0/{total_pages} pages"
#     )

#     try:

#         for i, page in enumerate(pdf):

#             # First try normal PDF text extraction
#             text = page.get_text("text")

#             extraction_type = "Normal extraction"

#             # Use OCR if extracted text is insufficient
#             if len(text.strip()) < 20:

#                 extraction_type = "OCR"

#                 pix = page.get_pixmap(
#                     matrix=fitz.Matrix(2.5, 2.5),
#                     colorspace=fitz.csRGB,
#                     alpha=False
#                 )

#                 image = Image.frombytes(
#                     "RGB",
#                     (pix.width, pix.height),
#                     pix.samples
#                 )

#                 text = pytesseract.image_to_string(
#                     image,
#                     lang="eng"
#                 )

#                 document = Document(
#                     page_content=text,
#                     metadata={
#                         "source": filename,
#                         "page": i,
#                         "file_type": "pdf"
#                     }
#                 )

#             documents.append(document)

#             # Update progress after every page
#             progress.progress(
#                 (i + 1) / total_pages,
#                 text=(
#                     f"{extraction_type}: "
#                     f"Page {i + 1}/{total_pages}"
#                 )
#             )

#     finally:
#         pdf.close()

#     progress.empty()

#     print(f"\nFile: {filename}")
#     print("Total pages:", len(documents))

#     for doc in documents[:5]:

#         print(
#             f"Page {doc.metadata['page']} | "
#             f"Characters: {len(doc.page_content.strip())} | "
#             f"Preview: {doc.page_content[:150]!r}"
#         )

#     return documents


def extract_pdf_documents(file_path, filename, ui=None, progress_callback=None):

    import re
    import math

    ui = ui or get_ui()

    documents = []

    pdf = fitz.open(str(file_path))
    total_pages = len(pdf)

    # --------------------------------------------------
    # STEP 1: Extract text blocks with their positions
    # --------------------------------------------------

    page_blocks = []
    header_candidates = []

    for page_index, page in enumerate(pdf):

        blocks = page.get_text("blocks")
        page_blocks.append(blocks)

        page_height = page.rect.height

        for block_index, block in enumerate(blocks):

            x0, y0, x1, y1, block_text = block[:5]

            # Ignore non-text blocks
            if len(block) >= 7 and block[6] != 0:
                continue

            block_text = block_text.strip()

            if not block_text:
                continue

            # Identify top and bottom page regions
            region = None

            if y0 <= page_height * 0.18:
                region = "top"

            elif y1 >= page_height * 0.82:
                region = "bottom"

            if region is None:
                continue

            # Normalize text to compare different header formats
            tokens = set(
                re.findall(r"[a-z]{2,}", block_text.lower())
            )

            # Ignore page numbers and very short text
            if len(tokens) < 3:
                continue

            header_candidates.append({
                "page": page_index,
                "block": block_index,
                "region": region,
                "position": ((y0 + y1) / 2) / page_height,
                "tokens": tokens,
                "text": block_text
            })

    # --------------------------------------------------
    # STEP 2: Find repeated headers and footers
    # --------------------------------------------------

    clusters = []

    for candidate in header_candidates:

        matched_cluster = None

        for cluster in clusters:

            representative = cluster[0]

            if candidate["region"] != representative["region"]:
                continue

            # Headers should appear at approximately the same position
            if abs(
                candidate["position"] - representative["position"]
            ) > 0.08:
                continue

            # Compare words without depending on their order
            intersection = (
                candidate["tokens"] & representative["tokens"]
            )

            union = (
                candidate["tokens"] | representative["tokens"]
            )

            similarity = len(intersection) / len(union)

            if similarity >= 0.75:
                matched_cluster = cluster
                break

        if matched_cluster is None:
            clusters.append([candidate])
        else:
            matched_cluster.append(candidate)

    # A repeated header must occur on multiple pages
    min_repeats = max(2, math.ceil(total_pages * 0.30))

    blocks_to_remove = set()

    for cluster in clusters:

        unique_pages = {
            item["page"] for item in cluster
        }

        if len(unique_pages) >= min_repeats:

            for item in cluster:

                blocks_to_remove.add(
                    (item["page"], item["block"])
                )

    print("\n========== REPEATED HEADERS / FOOTERS ==========")

    for cluster in clusters:

        unique_pages = {
            item["page"] for item in cluster
        }

        if len(unique_pages) >= min_repeats:

            print(
                f"Removed from {len(unique_pages)} pages: "
                f"{cluster[0]['text']!r}"
            )

    print("================================================")

    # --------------------------------------------------
    # STEP 3: Extract cleaned text and apply OCR if needed
    # --------------------------------------------------

    progress = ui.progress(
        0,
        text=f"Extracting PDF: 0/{total_pages} pages"
    )

    try:

        for i, page in enumerate(pdf):

            blocks = page_blocks[i]

            # Reconstruct text without repeated headers/footers
            remaining_blocks = []

            for block_index, block in enumerate(blocks):

                if (i, block_index) in blocks_to_remove:
                    continue

                # Ignore non-text blocks
                if len(block) >= 7 and block[6] != 0:
                    continue

                block_text = block[4].strip()

                if block_text:
                    remaining_blocks.append(block_text)

            text = "\n".join(remaining_blocks)

            extraction_type = "Normal extraction"

            # Use OCR if extracted text is insufficient
            if len(text.strip()) < 20:
                # print(f"[OCR] Processing page {i + 1}/{total_pages}")
                extraction_type = "OCR"

                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2.5, 2.5),
                    colorspace=fitz.csRGB,
                    alpha=False
                )

                image = Image.frombytes(
                    "RGB",
                    (pix.width, pix.height),
                    pix.samples
                )

                text = pytesseract.image_to_string(
                    image,
                    lang="eng"
                )
                # print(f"[OCR] Completed page {i + 1}/{total_pages}")
            document = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": i,
                    "file_type": "pdf"
                }
            )

            documents.append(document)

            if progress_callback:
                progress_callback({
                    "stage": f"Extracting {filename}: page {i + 1}/{total_pages}",
                    "progress": 10 + int(((i + 1) / total_pages) * 20)
                })

            progress.progress(
                (i + 1) / total_pages,
                text=(
                    f"{extraction_type}: "
                    f"Page {i + 1}/{total_pages}"
                )
            )

    finally:

        pdf.close()
        progress.empty()

    print(f"\nFile: {filename}")
    print("Total pages:", len(documents))

    for doc in documents[:5]:

        print(
            f"Page {doc.metadata['page']} | "
            f"Characters: {len(doc.page_content.strip())} | "
            f"Preview: {doc.page_content[:150]!r}"
        )

    return documents

def extract_txt_documents(file_path, filename, ui=None):

    ui = ui or get_ui()

    status = ui.status(
        f"Reading text file: {filename}",
        expanded=False
    )

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = filename
        document.metadata["file_type"] = "txt"

    status.update(
        label=f"Text extraction completed: {filename}",
        state="complete"
    )

    return documents

def extract_docx_documents(file_path, filename):

    word_doc = WordDocument(str(file_path))

    content = []

    # Extract paragraphs
    for paragraph in word_doc.paragraphs:

        text = paragraph.text.strip()

        if text:
            content.append(text)

    # Extract tables
    for table_index, table in enumerate(word_doc.tables, start=1):

        content.append(f"\nTable {table_index}:")

        rows = [
            [cell.text.strip() for cell in row.cells]
            for row in table.rows
        ]

        if not rows:
            continue

        headers = rows[0]

        for row_index, row in enumerate(rows[1:], start=2):

            row_content = []

            for column_index, value in enumerate(row):

                if not value:
                    continue

                header = (
                    headers[column_index]
                    if column_index < len(headers) and headers[column_index]
                    else f"Column {column_index + 1}"
                )

                row_content.append(f"{header}: {value}")

            if row_content:
                content.append(
                    f"Row {row_index}: " + ", ".join(row_content)
                )

    text = "\n".join(content)

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "file_type": "docx"
            }
        )
    ]



def extract_csv_documents(file_path, filename):

    documents = []

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            return documents

        for row_number, row in enumerate(reader, start=2):

            row_content = []

            for column, value in row.items():

                if column is None or value is None:
                    continue

                value = str(value).strip()

                if value:
                    row_content.append(f"{column}: {value}")

            if not row_content:
                continue

            documents.append(
                Document(
                    page_content="\n".join(row_content),
                    metadata={
                        "source": filename,
                        "file_type": "csv",
                        "row": row_number
                    }
                )
            )

    return documents

def extract_excel_documents(file_path, filename):

    documents = []

    workbook = load_workbook(
        filename=file_path,
        read_only=True,
        data_only=True
    )

    try:

        for worksheet in workbook.worksheets:

            rows = worksheet.iter_rows(values_only=True)

            headers = next(rows, None)

            if not headers:
                continue

            headers = [
                str(value).strip() if value is not None else f"Column {i + 1}"
                for i, value in enumerate(headers)
            ]

            for row_number, row in enumerate(rows, start=2):

                row_content = []

                for column_index, value in enumerate(row):

                    if value is None:
                        continue

                    value = str(value).strip()

                    if not value:
                        continue

                    header = (
                        headers[column_index]
                        if column_index < len(headers)
                        else f"Column {column_index + 1}"
                    )

                    row_content.append(f"{header}: {value}")

                if not row_content:
                    continue

                documents.append(
                    Document(
                        page_content=(
                            f"Worksheet: {worksheet.title}\n"
                            + "\n".join(row_content)
                        ),
                        metadata={
                            "source": filename,
                            "file_type": "xlsx",
                            "sheet": worksheet.title,
                            "row": row_number
                        }
                    )
                )

    finally:
        workbook.close()

    return documents

def load_document(file_path, filename=None, ui=None, progress_callback=None):

    ui = ui or get_ui()

    file_path = Path(file_path)

    if filename is None:
        filename = file_path.name

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":

        documents = extract_pdf_documents(
            file_path,
            filename,
            ui=ui,
            progress_callback=progress_callback
        )

    elif suffix == ".txt":

        documents = extract_txt_documents(
            file_path,
            filename,
            ui=ui
        )

    elif suffix == ".docx":

        documents = extract_docx_documents(
            file_path,
            filename
        )

    elif suffix == ".csv":

        documents = extract_csv_documents(
            file_path,
            filename
        )

    elif suffix == ".xlsx":

        documents = extract_excel_documents(
            file_path,
            filename
        )

    else:

        raise ValueError(
            f"Unsupported file type: {suffix}"
        )

    # return documents

    # Assign one unique ID to each uploaded document.
    document_id = uuid4().hex

    for document in documents:

        document.metadata["document_id"] = document_id

        # Keep the existing zero-based page numbering.
        # Non-paginated formats receive None.
        document.metadata.setdefault("page", None)

        document.metadata.setdefault("source", filename)
        document.metadata.setdefault("file_type", suffix.lstrip("."))

    return documents



# --------------------------------------------------
# 4. CREATE RAG
# --------------------------------------------------

# def create_rag(name, uploaded_files, show_ui=True):

#     ui = get_ui(show_ui)

#     rag_id = uuid4().hex

#     rag_dir = RAGS_DIR / rag_id
#     rag_dir.mkdir(parents=True, exist_ok=False)

#     documents = []

#     overall_status = ui.status(
#         "Preparing your documents...",
#         expanded=True
#     )

#     rag_id = uuid4().hex

#     rag_dir = RAGS_DIR / rag_id
#     rag_dir.mkdir(parents=True, exist_ok=False)

#     documents = []

#     # Overall processing status
#     overall_status = st.status(
#         "Preparing your documents...",
#         expanded=True
#     )

#     try:

#         # ------------------------------------------
#         # STEP 1: SAVE AND EXTRACT FILES
#         # ------------------------------------------

#         overall_status.update(
#             label="Step 1/5: Saving and extracting documents..."
#         )

#         save_progress = st.progress(
#             0,
#             text="Saving uploaded files..."
#         )

#         for i, uploaded_file in enumerate(uploaded_files):

#             filename = Path(uploaded_file.name).name

#             file_path = rag_dir / filename

#             file_path.write_bytes(
#                 uploaded_file.getvalue()
#             )
#             file_documents = load_document(
#                 file_path,
#                 filename
#             )

#             documents.extend(file_documents)

#             save_progress.progress(
#                 (i + 1) / len(uploaded_files),
#                 text=(
#                     f"Files processed: "
#                     f"{i + 1}/{len(uploaded_files)}"
#                 )
#             )

#         save_progress.empty()

#         if not documents:
#             raise ValueError(
#                 "No documents were extracted."
#             )

#         # ------------------------------------------
#         # STEP 2: CHUNKING
#         # ------------------------------------------

#         overall_status.update(
#             label="Step 2/5: Splitting documents into chunks..."
#         )

#         chunk_status = st.status(
#             "Creating text chunks...",
#             expanded=False
#         )

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=600,
#             chunk_overlap=100
#         )

#         chunks = splitter.split_documents(documents)

#         chunk_status.update(
#             label=f"Chunking completed: {len(chunks)} chunks",
#             state="complete"
#         )

#         print("\n--- CHUNKING DEBUG ---")
#         print("Total documents:", len(documents))
#         print("Total chunks:", len(chunks))

#         for i, chunk in enumerate(chunks[:5]):

#             print(
#                 f"Chunk {i + 1} | "
#                 f"Length: {len(chunk.page_content)} | "
#                 f"Content: {chunk.page_content[:200]!r}"
#             )

#         if not chunks:
#             raise ValueError(
#                 "No text chunks were created."
#             )

#         # ------------------------------------------
#         # STEP 3: GENERATE EMBEDDINGS
#         # ------------------------------------------

#         overall_status.update(
#             label="Step 3/5: Generating embeddings..."
#         )

#         embedding_progress = st.progress(
#             0,
#             text="Starting embedding generation..."
#         )

#         batch_size = 32

#         text_embeddings = []

#         total_chunks = len(chunks)

#         for start in range(0, total_chunks, batch_size):

#             batch = chunks[start:start + batch_size]

#             texts = [
#                 chunk.page_content
#                 for chunk in batch
#             ]

#             vectors = embeddings.embed_documents(texts)

#             # Pair each text with its embedding
#             text_embeddings.extend(
#                 zip(texts, vectors)
#             )

#             processed = min(
#                 start + len(batch),
#                 total_chunks
#             )

#             embedding_progress.progress(
#                 processed / total_chunks,
#                 text=(
#                     f"Generating embeddings: "
#                     f"{processed}/{total_chunks} chunks"
#                 )
#             )

#         embedding_progress.empty()

#         # ------------------------------------------
#         # STEP 4: BUILD FAISS INDEX
#         # ------------------------------------------

#         overall_status.update(
#             label="Step 4/5: Building FAISS vector index..."
#         )

#         index_progress = st.progress(
#             0,
#             text="Building FAISS index..."
#         )

#         vector_store = FAISS.from_embeddings(
#             text_embeddings=text_embeddings,
#             embedding=embeddings,
#             metadatas=[
#                 chunk.metadata for chunk in chunks
#             ]
#         )

#         index_progress.progress(
#             1.0,
#             text="FAISS index created successfully"
#         )

#         # ------------------------------------------
#         # STEP 5: SAVE INDEX AND CATALOG
#         # ------------------------------------------

#         overall_status.update(
#             label="Step 5/5: Saving vector index..."
#         )

#         save_progress = st.progress(
#             0,
#             text="Saving FAISS index..."
#         )

#         index_path = rag_dir / "faiss_index"

#         vector_store.save_local(str(index_path))

#         save_progress.progress(
#             0.7,
#             text="Saving RAG metadata..."
#         )

#         catalog = load_catalog()

#         catalog[rag_id] = {
#             "name": name.strip(),
#             "index_path": str(
#                 index_path.relative_to(BASE_DIR)
#             ),
#             "files": [
#                 Path(file.name).name
#                 for file in uploaded_files
#             ]
#         }

#         save_catalog(catalog)

#         save_progress.progress(
#             1.0,
#             text="Saving completed"
#         )

#         overall_status.update(
#             label="All processing completed successfully!",
#             state="complete",
#             expanded=False
#         )

#         st.success(
#             f"RAG created successfully! "
#             f"{len(chunks)} chunks indexed."
#         )

#         return rag_id

#     except Exception as e:

#         overall_status.update(
#             label="Document processing failed",
#             state="error",
#             expanded=True
#         )

#         raise e




def detect_heading(line):
    """
    Detect common document heading patterns.
    Returns (heading_depth, heading_title) or None.
    """

    line = line.strip()

    if not line:
        return None

    # Markdown headings: #, ##, ###
    match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)

    if match:
        return len(match.group(1)), match.group(2).strip()

    # Named headings: Chapter 1, Part II, Section 2, etc.
    match = re.match(
        r"^(chapter|part|section|appendix|article)\s+"
        r"([A-Z0-9IVXLCDM.-]+)"
        r"(?:\s*[:\-–]\s*|\s+)?(.*)$",
        line,
        re.IGNORECASE
    )

    if match:
        heading_type = match.group(1).lower()

        depth = (
            1
            if heading_type in ("chapter", "part", "appendix")
            else 2
        )

        return depth, line

    # Hierarchical numbered headings: 1.1, 1.1.1, etc.
    match = re.match(
        r"^(\d+(?:\.\d+){1,5})[.)]?\s+(.+)$",
        line
    )

    if match:
        number = match.group(1)
        depth = len(number.split("."))

        return depth, line

    # Single-level numbered headings: 1. Introduction
    # Only accept short title-like lines, not numbered sentences.
    match = re.match(
        r"^(\d+)[.)]\s+(.+)$",
        line
    )

    if match:
        title = match.group(2).strip()
        words = title.split()

        if (
            len(words) <= 8
            and not re.search(r"[.!?;:]$", title)
        ):
            return 2, line

    # Roman numbered headings: I. Introduction, II. Background
    match = re.match(
        r"^([IVXLCDM]+)[.)]\s+(.+)$",
        line,
        re.IGNORECASE
    )

    if match:
        title = match.group(2).strip()
        words = title.split()

        if (
            len(words) <= 8
            and not re.search(r"[.!?;:]$", title)
        ):
            return 1, line

    return None



def is_numbered_list_item(lines, index):

    pattern = re.compile(r"^\s*(\d+)[.)]\s+.+$")

    current = pattern.match(lines[index])

    if not current:
        return False

    current_number = int(current.group(1))

    # Check the next non-empty line
    next_index = index + 1

    while next_index < len(lines) and not lines[next_index].strip():
        next_index += 1

    if next_index < len(lines):
        next_match = pattern.match(lines[next_index])

        if next_match and int(next_match.group(1)) == current_number + 1:
            return True

    # Check the previous non-empty line
    previous_index = index - 1

    while previous_index >= 0 and not lines[previous_index].strip():
        previous_index -= 1

    if previous_index >= 0:
        previous_match = pattern.match(lines[previous_index])

        if previous_match and int(previous_match.group(1)) == current_number - 1:
            return True

    return False



def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = []
    document_chunk_counts = {}
    parent_sections = {}

    # Maintain heading hierarchy separately for each document.
    document_sections = {}


    def get_parent_key(document_id, section_path, metadata):

        # CSV and Excel: each row is an independent parent.
        if metadata.get("row") is not None:
            return (
                document_id,
                section_path,
                metadata.get("sheet"),
                metadata.get("row")
            )

        # PDF without detected headings: use each page as a parent.
        if (
            section_path == "Document body"
            and metadata.get("page") is not None
        ):
            return (
                document_id,
                section_path,
                metadata.get("page")
            )

        # Normal documents: one parent per section.
        return (document_id, section_path)

   

    line_counts = Counter()

    for document in documents:
        unique_lines = {
            line.strip()
            for line in document.page_content.splitlines()
            if line.strip()
        }
        line_counts.update(unique_lines)

    print("\n========== REPEATED PDF LINES ==========")

    for line, count in line_counts.most_common(20):
        if count >= 2:
            print(f"{count} times: {line}")

    print("========================================")



    for document in documents:

        metadata = document.metadata or {}

        document_id = metadata.get("document_id")

        if not document_id:
            document_id = uuid4().hex

        if document_id not in document_sections:
            document_sections[document_id] = []

        heading_stack = document_sections[document_id]

        lines = document.page_content.splitlines()

        section_blocks = []

        current_lines = []

        current_title = (
            heading_stack[-1]
            if heading_stack
            else "Document body"
        )

        def flush_section():

            if not current_lines:
                return

            section_blocks.append({
                "content": "\n".join(current_lines),
                "section_title": current_title,
                "parent_section": (
                    heading_stack[-2]
                    if len(heading_stack) >= 2
                    else None
                ),
                "section_path": (
                    " > ".join(heading_stack)
                    if heading_stack
                    else "Document body"
                )
            })

            current_lines.clear()

        # for line in lines:

        line_index = 0

        while line_index < len(lines):

            line = lines[line_index]
            heading = detect_heading(line)

            # Avoid interpreting consecutive numbered list items as headings.
            if heading and is_numbered_list_item(lines, line_index):
                heading = None
  
            heading_content = line

            # Handle PDF headings where the number and title
            # are extracted on separate lines, e.g.:
            # 1.1
            # What is Machine Learning?

            if (
                not heading
                and re.fullmatch(r"\d+(?:\.\d+)+", line.strip())
            ):

                next_index = line_index + 1

                while (
                    next_index < len(lines)
                    and not lines[next_index].strip()
                ):
                    next_index += 1

                if next_index < len(lines):

                    next_line = lines[next_index].strip()

                    combined_heading = f"{line.strip()} {next_line}"

                    heading = detect_heading(combined_heading)

                    if heading:
                        heading_content = combined_heading
                        line_index = next_index

            if heading:

                flush_section()

                depth, title = heading

                del heading_stack[depth - 1:]

                heading_stack.append(title)

                current_title = title

                current_lines.append(heading_content)

            else:

                current_lines.append(line)

            line_index += 1


        flush_section()

        # Split each section independently.
        # Split each section independently.
        for section in section_blocks:

            parent_key = get_parent_key(
                document_id,
                section["section_path"],
                metadata
            )

            parent_entry = parent_sections.setdefault(
                parent_key,
                {
                    "parent_id": uuid4().hex,
                    "content": []
                }
            )

            # Collect original section text across pages.
            parent_entry["content"].append(
                section["content"]
            )

            section_document = Document(
                page_content=section["content"],
                metadata=metadata.copy()
            )

            section_chunks = splitter.split_documents(
                [section_document]
            )

            for chunk in section_chunks:

                # chunk_number = len(chunks)
                chunk_number = document_chunk_counts.get(document_id, 0)

                document_chunk_counts[document_id] = chunk_number + 1

                section_path = section["section_path"]

                # Add heading context to searchable text.
                chunk.page_content = (
                    f"Section context: {section_path}\n\n"
                    f"{chunk.page_content}"
                )

                # Preserve source metadata and add structure.
                chunk.metadata.update({
                    "document_id": document_id,
                    "chunk_id": uuid4().hex,
                    "chunk_index": chunk_number,
                    "section_title": section["section_title"],
                    "parent_section": section["parent_section"],
                    "section_path": section_path
                })

                chunks.append(chunk)

    print("\n========== STRUCTURE-AWARE CHUNKING ==========")
    # Attach the complete parent section to every child chunk.
    for chunk in chunks:

        metadata = chunk.metadata

        parent_key = get_parent_key(
            metadata["document_id"],
            metadata["section_path"],
            metadata
        )

        parent = parent_sections[parent_key]

        chunk.metadata.update({
            "parent_id": parent["parent_id"],
            "parent_content": "\n\n".join(parent["content"])
        })

    print("Input documents:", len(documents))
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:10], start=1):

        print(f"\nChunk {i}")
        print("Document ID:", chunk.metadata["document_id"])
        print("Page:", chunk.metadata.get("page"))
        print("Section:", chunk.metadata["section_title"])
        print("Parent:", chunk.metadata["parent_section"])
        print("Path:", chunk.metadata["section_path"])
        print("Content:", chunk.page_content[:250])

    print("==============================================")

    return chunks





def create_rag(name, uploaded_files, show_ui=True, progress_callback=None):

    ui = get_ui(show_ui)
    def report_progress(stage, percentage):
        if progress_callback:
            progress_callback({
                "stage": stage,
                "progress": percentage
            })
    rag_id = uuid4().hex

    rag_dir = RAGS_DIR / rag_id
    rag_dir.mkdir(parents=True, exist_ok=False)

    documents = []

    overall_status = ui.status(
        "Preparing your documents...",
        expanded=True
    )

    try:
        report_progress("Uploading and extracting documents", 10)
        # ------------------------------------------
        # STEP 1: SAVE AND EXTRACT FILES
        # ------------------------------------------

        overall_status.update(
            label="Step 1/5: Saving and extracting documents..."
        )

        save_progress = ui.progress(
            0,
            text="Saving uploaded files..."
        )

        for i, uploaded_file in enumerate(uploaded_files):

            filename = Path(uploaded_file.name).name

            file_path = rag_dir / filename

            file_path.write_bytes(
                uploaded_file.getvalue()
            )

            file_documents = load_document(
                file_path,
                filename,
                ui=ui,
                progress_callback=progress_callback
            )
            documents.extend(file_documents)

            for doc in file_documents:
                if "classification" in doc.page_content.lower():
                    print("\n========== RAW PDF TEXT ==========")
                    print("Page:", doc.metadata.get("page"))
                    print(doc.page_content)


            save_progress.progress(
                (i + 1) / len(uploaded_files),
                text=(
                    f"Files processed: "
                    f"{i + 1}/{len(uploaded_files)}"
                )
            )

        save_progress.empty()

        if not documents:
            raise ValueError(
                "No documents were extracted."
            )
        report_progress("Splitting documents into chunks", 30)
        # ------------------------------------------
        # STEP 2: CHUNKING
        # ------------------------------------------

        overall_status.update(
            label="Step 2/5: Splitting documents into chunks..."
        )

        chunk_status = ui.status(
            "Creating text chunks...",
            expanded=False
        )

        chunks = chunk_documents(documents)


        for i, chunk in enumerate(chunks):
            if "classification" in chunk.page_content.lower():
                print(f"\n========== CLASSIFICATION CHUNK {i} ==========")
                print("Page:", chunk.metadata.get("page"))
                print("Section:", chunk.metadata.get("section_title"))
                print("Path:", chunk.metadata.get("section_path"))
                print("Length:", len(chunk.page_content))
                print(chunk.page_content)


        for i, chunk in enumerate(chunks):
            if chunk.metadata.get("page") in [2, 3]:
                print(f"\n========== CHUNK {i} ==========")
                print("Page:", chunk.metadata.get("page"))
                print("Section:", chunk.metadata.get("section_title"))
                print("Path:", chunk.metadata.get("section_path"))
                print(chunk.page_content)

        chunk_status.update(
            label=f"Chunking completed: {len(chunks)} chunks",
            state="complete"
        )

        print("\n--- CHUNKING DEBUG ---")
        print("Total documents:", len(documents))
        print("Total chunks:", len(chunks))

        for i, chunk in enumerate(chunks[:5]):

            print(
                f"Chunk {i + 1} | "
                f"Length: {len(chunk.page_content)} | "
                f"Content: {chunk.page_content[:200]!r}"
            )

        if not chunks:
            raise ValueError(
                "No text chunks were created."
            )

        report_progress("Generating embeddings", 50)
        # ------------------------------------------
        # STEP 3: GENERATE EMBEDDINGS
        # ------------------------------------------

        overall_status.update(
            label="Step 3/5: Generating embeddings..."
        )

        embedding_progress = ui.progress(
            0,
            text="Starting embedding generation..."
        )

        batch_size = 32

        text_embeddings = []

        total_chunks = len(chunks)

        for start in range(0, total_chunks, batch_size):

            batch = chunks[start:start + batch_size]

            texts = [
                chunk.page_content
                for chunk in batch
            ]

            vectors = embeddings.embed_documents(texts)

            text_embeddings.extend(
                zip(texts, vectors)
            )

            processed = min(
                start + len(batch),
                total_chunks
            )

            embedding_progress.progress(
                processed / total_chunks,
                text=(
                    f"Generating embeddings: "
                    f"{processed}/{total_chunks} chunks"
                )
            )

        embedding_progress.empty()

        # ------------------------------------------
        # STEP 4: BUILD FAISS INDEX
        # ------------------------------------------
        report_progress("Building FAISS index", 80)

        overall_status.update(
            label="Step 4/5: Building FAISS vector index..."
        )

        index_progress = ui.progress(
            0,
            text="Building FAISS index..."
        )

        vector_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=embeddings,
            metadatas=[
                chunk.metadata for chunk in chunks
            ]
        )

        index_progress.progress(
            1.0,
            text="FAISS index created successfully"
        )

        # ------------------------------------------
        # STEP 5: SAVE INDEX AND CATALOG
        # ------------------------------------------
        report_progress("Saving vector index", 95)

        overall_status.update(
            label="Step 5/5: Saving vector index..."
        )

        save_progress = ui.progress(
            0,
            text="Saving FAISS index..."
        )

        index_path = rag_dir / "faiss_index"

        vector_store.save_local(str(index_path))

        save_progress.progress(
            0.7,
            text="Saving RAG metadata..."
        )

        catalog = load_catalog()

        catalog[rag_id] = {
            "name": name.strip(),
            "index_path": str(
                index_path.relative_to(BASE_DIR)
            ),
            "files": [
                Path(file.name).name
                for file in uploaded_files
            ]
        }

        save_catalog(catalog)

        save_progress.progress(
            1.0,
            text="Saving completed"
        )

        overall_status.update(
            label="All processing completed successfully!",
            state="complete",
            expanded=False
        )

        ui.success(
            f"RAG created successfully! "
            f"{len(chunks)} chunks indexed."
        )
        report_progress("Project created successfully", 100)
        
        return rag_id

    except Exception as e:

        overall_status.update(
            label="Document processing failed",
            state="error",
            expanded=True
        )

        # Remove incomplete project files after failure
        import shutil
        shutil.rmtree(rag_dir, ignore_errors=True)

        raise e


# --------------------------------------------------
# 5. LOAD EXISTING VECTOR STORE
# --------------------------------------------------

def load_vector_store(rag_id, show_ui=True):

    ui = get_ui(show_ui)

    catalog = load_catalog()

    if rag_id not in catalog:
        raise ValueError(
            "The requested RAG does not exist."
        )

    index_path = BASE_DIR / catalog[rag_id]["index_path"]

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    with ui.spinner("Loading FAISS vector index..."):

        vector_store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("\n--- FAISS DEBUG ---")
    print("Loaded RAG ID:", rag_id)
    print("Loaded index path:", index_path)
    print(
        "Total indexed chunks:",
        len(vector_store.docstore._dict)
    )

    return vector_store
