"""Index PDFs, text, Markdown, and bounded CSV content into local ChromaDB."""

import chromadb
from pathlib import Path
from threading import Lock
from sentence_transformers import SentenceTransformer

from rag import create_chunks, load_documents


DATABASE_FOLDER = "database"
COLLECTION_NAME = "research_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_LOCK = Lock()


def index_documents(embedding_model=None, return_details=False):
    with INDEX_LOCK:
        client = chromadb.PersistentClient(path=DATABASE_FOLDER)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

        documents = load_documents()
        chunks = create_chunks(documents)
        source_folder = Path("documents")
        present_sources = {
            str(path.relative_to(source_folder))
            for path in source_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md", ".csv"}
        } if source_folder.exists() else set()
        loaded_sources = {document["filename"] for document in documents}

        ids = []
        texts = []
        metadatas = []
        for chunk in chunks:
            filename = chunk["filename"]
            ids.append(f"{filename}_page_{chunk['page']}_chunk_{chunk['chunk']}")
            texts.append(chunk["text"])
            metadata = {"filename": filename, "page": chunk["page"], "chunk": chunk["chunk"]}
            if chunk.get("source_url"):
                metadata["source_url"] = chunk["source_url"]
            metadatas.append(metadata)

        existing = collection.get(include=["documents", "metadatas"])
        old_documents = dict(zip(existing["ids"], existing["documents"]))
        old_metadata = dict(zip(existing["ids"], existing["metadatas"]))
        desired_ids = set(ids)
        changed = [index for index, chunk_id in enumerate(ids) if old_documents.get(chunk_id) != texts[index]]

        if changed:
            if embedding_model is None:
                print("Loading embedding model...")
                embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            changed_texts = [texts[index] for index in changed]
            embeddings = embedding_model.encode(changed_texts, show_progress_bar=True)
            collection.upsert(
                ids=[ids[index] for index in changed],
                documents=changed_texts,
                embeddings=embeddings.tolist(),
                metadatas=[metadatas[index] for index in changed],
            )

        metadata_updates = [
            index for index, chunk_id in enumerate(ids)
            if chunk_id in old_metadata and old_documents.get(chunk_id) == texts[index]
            and old_metadata[chunk_id] != metadatas[index]
        ]
        if metadata_updates:
            collection.update(
                ids=[ids[index] for index in metadata_updates],
                metadatas=[metadatas[index] for index in metadata_updates],
            )

        stale_ids = [
            chunk_id for chunk_id, metadata in zip(existing["ids"], existing["metadatas"])
            if (metadata or {}).get("filename") not in present_sources
            or ((metadata or {}).get("filename") in loaded_sources and chunk_id not in desired_ids)
        ]
        if stale_ids:
            collection.delete(ids=stale_ids)

        print(f"Indexed chunks: {collection.count()} ({len(changed)} new or changed)")
        indexed_count = collection.count()
        if return_details:
            return {
                "indexed_chunks": indexed_count,
                "new_or_changed_chunks": len(changed),
                "removed_stale_chunks": len(stale_ids),
            }
        return indexed_count


if __name__ == "__main__":
    index_documents()
