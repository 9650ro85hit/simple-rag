from langchain_core.documents import Document
from rag_storage import chunk_documents

test_document = Document(
    page_content="""
Chapter 1: Introduction

1.1 Background

This section explains the background of the system.

1.2 Objectives

This section explains the objectives.

Chapter 2: Architecture

2.1 Components

This section explains the system components.
""",
    metadata={
        "document_id": "test_document",
        "source": "test.txt",
        "page": 0
    }
)

# Actually execute the chunking function
test_chunks = chunk_documents([test_document])

for i, chunk in enumerate(test_chunks):

    print(f"\n========== CHUNK {i + 1} ==========")

    print("Section:", chunk.metadata.get("section_title"))
    print("Parent:", chunk.metadata.get("parent_section"))
    print("Path:", chunk.metadata.get("section_path"))
    print("Page:", chunk.metadata.get("page"))

    print("\nContent:")
    print(chunk.page_content)