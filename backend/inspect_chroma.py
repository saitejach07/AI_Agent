from app.config import get_settings
from app.rag.vector_store import get_vector_store


settings = get_settings()

vector_store = get_vector_store(settings)

raw = vector_store.get(
    include=["documents", "metadatas", "embeddings"]
)

for i, document in enumerate(raw["documents"]):
    print("\n" + "=" * 80)
    print(f"CHUNK {i}")
    print("=" * 80)

    print("\nCHROMA ID:")
    print(raw["ids"][i])

    print("\nDOCUMENT:")
    print(document)

    print("\nMETADATA:")
    print(raw["metadatas"][i])

    print("\nEMBEDDING:")
    embedding = raw["embeddings"][i]
    print(embedding)
    print(f"Dimensions: {len(embedding)}")