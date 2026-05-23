# ingest.py
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# ── Embeddings model (downloads once, runs locally) ───────
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── ChromaDB client (creates a local folder ./chroma_db) ──
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)

# ── Load your documents ───────────────────────────────────
# Option A: single text file
loader = TextLoader("docs/sample.txt")

# Option B: entire folder of .txt files
# loader = DirectoryLoader("docs/", glob="**/*.txt")

documents = loader.load()

# ── Chunk them ────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # characters per chunk
    chunk_overlap=50,     # overlap so context isn't lost at boundaries
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(documents)

# ── Embed and store ───────────────────────────────────────
for i, chunk in enumerate(chunks):
    embedding = embeddings_model.embed_query(chunk.page_content)
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk.page_content],
        metadatas=[{"source": chunk.metadata.get("source", "unknown")}]
    )

print(f"✅ Ingested {len(chunks)} chunks into ChromaDB")