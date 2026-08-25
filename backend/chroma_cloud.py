import os
import chromadb
from dotenv import load_dotenv


load_dotenv()


EMBEDDING_MODEL = "text-embedding-3-small"


client = chromadb.CloudClient(
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"],
    api_key=os.environ["CHROMA_API_KEY"],
)


print("Connected to Chroma Cloud!")
print(f"Application embedding model: {EMBEDDING_MODEL}")


try:
    collection = client.get_collection("rag_documents")

    print("Collection 'rag_documents' already exists.")
    print("Using existing collection.")

except Exception:
    collection = client.create_collection(
        name="rag_documents",
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )

    print("Collection 'rag_documents' did not exist.")
    print("Created collection with cosine similarity.")


print(f"Embedding model used by application: {EMBEDDING_MODEL}")

print("Collection configuration:")
print(collection.configuration)

print("Collection metadata:")
print(collection.metadata)