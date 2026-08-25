from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.rag.vector_store import get_vector_store
from app.schemas import DocumentInfo, UploadResult
from fastapi import HTTPException, status


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


# def _safe_upload_name(filename: str) -> str:
#     return Path(filename).name.replace(" ", "_")


async def ingest_upload(file: UploadFile, settings: Settings) -> UploadResult:
    """
    Ingests one uploaded file.

    Flow:
    1. Validate file type.
    2. Save uploaded file locally.
    3. Load document text.
    4. Split text into chunks.
    5. Add chunk metadata.
    6. Store chunks in ChromaDB.
    """

    original_name = file.filename or "document"
    suffix = Path(original_name).suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Upload PDF, TXT, or MD files.",
        )

    source_id = uuid4().hex
    #local storage 

    #stored_name = f"{source_id}_{_safe_upload_name(original_name)}"
    # target_path = settings.upload_dir / stored_name

    # target_path.write_bytes(await file.read())

    # documents = _load_documents(
    #     path=target_path,
    #     filename=original_name,
    #     source_id=source_id,
    # )

    #using RAM
    file_bytes = await file.read()

    documents = _load_documents(
        file_bytes=file_bytes,
        filename=original_name,
        source_id=source_id,
    )

    chunks = _split_documents(
        documents=documents,
        settings=settings,
    )

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    vector_store = get_vector_store(settings)
    vector_store.add_documents(chunks)  # storing in chromadb

    return UploadResult(
        source_id=source_id,
        filename=original_name,
        chunks=len(chunks),
    )


def list_indexed_documents(settings: Settings) -> list[DocumentInfo]:
    """
    Lists documents already indexed in ChromaDB.

    ChromaDB stores chunks, so we group chunks back by source_id.
    """

    vector_store = get_vector_store(settings)

    raw = vector_store.get(include=["metadatas"])

    grouped: dict[str, DocumentInfo] = {}

    for metadata in raw.get("metadatas", []):
        if not metadata:
            continue

        source_id = str(metadata.get("source_id", "unknown"))
        filename = str(metadata.get("filename", "unknown"))

        if source_id not in grouped:
            grouped[source_id] = DocumentInfo(
                source_id=source_id,
                filename=filename,
                chunks=0,
            )

        grouped[source_id].chunks += 1

    return sorted(
        grouped.values(),
        key=lambda item: item.filename.lower(),
    )

# local storage
# def _load_documents(
#     file_bytes: bytes,
#     path: Path,
#     filename: str,
#     source_id: str,
# ) -> list[Document]:
#     """
#     Loads text from PDF, TXT, or MD files.
#     """
#     suffix = path.suffix.lower()

#     if suffix == ".pdf":
#         docs = PyPDFLoader(str(path)).load()
#     else:
#         docs = TextLoader(str(path), encoding="utf-8").load()

#     for doc in docs:
#         doc.metadata.update(
#             {
#                 "source_id": source_id,
#                 "filename": filename,
#                 "page": doc.metadata.get("page"),
#             }
#         )

#     return docs

def _load_documents(
    file_bytes: bytes,
    filename: str,
    source_id: str,
) -> list[Document]:
    """
    Loads text from PDF, TXT, or MD files directly from memory.

    The uploaded file is not saved to local disk.
    """

    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        import io
        from pypdf import PdfReader

        pdf = PdfReader(io.BytesIO(file_bytes))

        docs = []

        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_number,
                    },
                )
            )

    else:
        text = file_bytes.decode("utf-8")

        docs = [
            Document(
                page_content=text,
                metadata={},
            )
        ]

    for doc in docs:
        doc.metadata.update(
            {
                "source_id": source_id,
                "filename": filename,
                "page": doc.metadata.get("page"),
            }
        )

    return docs


def _split_documents(
    documents: list[Document],
    settings: Settings,
) -> list[Document]:
    """
    Splits loaded documents into smaller chunks.

    Chunking happens before embeddings.
    When chunks are added to ChromaDB, LangChain uses the embedding model
    from vector_store.py to embed each chunk.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)

def delete_indexed_document(
    filename: str,
    settings: Settings,
) -> None:
    vector_store = get_vector_store(settings)

    # Find all chunks belonging to this exact filename.
    results = vector_store.get(
        where={"filename": filename},
        include=["metadatas"],
    )

    matching_ids = results.get("ids", [])

    # Nothing found with this exact filename.
    if not matching_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document found with filename '{filename}'",
        )

    # Delete all chunks belonging to the document.
    vector_store.delete(
        ids=matching_ids,
    ) 

