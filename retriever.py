# retriever.py
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# ── Load the same model and DB ────────────────────────────
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def retrieve_docs(query: str, top_k: int = 3) -> list[str]:
    """
    Embed the query, search ChromaDB, return top_k relevant chunks.
    """
    query_embedding = embeddings_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    docs = results["documents"][0]       # list of chunk strings
    distances = results["distances"][0]  # lower = more similar (cosine)

    # Filter out low-relevance results (distance > 0.7 = not very similar)
    filtered = [
        doc for doc, dist in zip(docs, distances)
        if dist < 0.7
    ]

    return filtered if filtered else ["No relevant documents found."]