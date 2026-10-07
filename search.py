import chromadb

from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

DATABASE_FOLDER = "database"

COLLECTION_NAME = "research_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded.")


# ============================================================
# OPEN DATABASE
# ============================================================

client = chromadb.PersistentClient(
    path=DATABASE_FOLDER
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"Database contains {collection.count()} chunks.")


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_documents(query, number_of_results=5):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=number_of_results
    )

    return results


# ============================================================
# INTERACTIVE SEARCH
# ============================================================

print()
print("=" * 60)
print("AKITO RESEARCH DOCUMENT SEARCH")
print("=" * 60)
print("Type 'exit' to quit.")
print()


while True:

    query = input("Search: ")

    if query.lower() == "exit":
        break

    results = search_documents(query)

    print()
    print("=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):

        metadata = metadatas[i]

        print()
        print(f"Result {i + 1}")
        print("-" * 60)

        print(
            f"Source: {metadata['filename']}"
        )

        print(
            f"Page: {metadata['page']}"
        )

        print(
            f"Chunk: {metadata['chunk']}"
        )

        print(
            f"Distance: {distances[i]:.4f}"
        )

        print()
        print(documents[i])

    print()