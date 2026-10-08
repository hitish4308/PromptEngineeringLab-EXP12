"""
Experiment 12 - RAG Data Indexing Pipeline

Loads sample text, chunks it, embeds it with NVIDIA NIM, stores it
in a FAISS vector store, and retrieves top-2 chunks for a query.

Uses NVIDIAEmbeddings instead of OpenAIEmbeddings because this
environment is configured for NVIDIA NIM.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ---------- Setup ----------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise SystemExit("NVIDIA_API_KEY not found. Create a .env file.")

EMBED_MODEL = "nvidia/nemotron-3-embed-1b"   # stable NVIDIA embedding model


# ---------- Sample Text ----------
SAMPLE_TEXT = """
Climate change refers to long-term shifts in temperatures and weather
patterns. These shifts may be natural, but since the 1800s, human activities
have been the main driver of climate change, primarily due to the burning of
fossil fuels like coal, oil, and gas, which produces heat-trapping gases.

Rising global temperatures cause sea levels to rise due to melting polar ice
caps and thermal expansion of seawater. Coastal cities face flooding risks
and saltwater intrusion into freshwater supplies.

More frequent and intense storms, droughts, and heatwaves are also linked to
climate change. These extremes stress agriculture, damage infrastructure,
and displace communities, with vulnerable populations hit hardest.

Scientists agree that reducing greenhouse gas emissions is essential.
Renewable energy, energy efficiency, and sustainable land use are key
mitigation strategies.
"""


# ---------- Query ----------
QUERY = "What are the effects of climate change?"


# ---------- Main ----------
if __name__ == "__main__":
    print("=" * 72)
    print("  EXPERIMENT 12 - RAG INDEXING PIPELINE")
    print("=" * 72)
    print(f"Embedding model: {EMBED_MODEL}")
    print()

    # ---------- Step 1: Chunk ----------
    print("Step 1: Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = splitter.create_documents([SAMPLE_TEXT])
    print(f"  Chunk size: 300 | Overlap: 30")
    print(f"  Number of chunks: {len(documents)}")
    for i, doc in enumerate(documents):
        print(f"    Chunk {i+1}: {len(doc.page_content)} chars")
    print()

    # ---------- Step 2: Embed ----------
    print("Step 2: Initializing embedding model...")
    embeddings = NVIDIAEmbeddings(model=EMBED_MODEL, model_type="passage")
    print("  Model ready.")
    print()

    # ---------- Step 3: Store in FAISS ----------
    print("Step 3: Building FAISS vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print(f"  Vector store created with {len(documents)} chunks.")
    print()

    # ---------- Step 4: Query ----------
    print("=" * 72)
    print(f"  QUERY: {QUERY}")
    print("=" * 72)
    print()

    # Use passage embeddings for the query - NVIDIA models expect different
    # input types for query vs document, but for this demo we use the
    # query embedding method which handles that internally.
    results = vectorstore.similarity_search_with_relevance_scores(QUERY, k=2)

    print(f"Top {len(results)} chunks:\n")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"  --- Result {i} ---")
        print(f"  Score: {score:.4f}")
        print(f"  Chunk: {doc.page_content.strip()[:200]}...")
        print()

    # ---------- Summary ----------
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print("""
RAG indexing pipeline:

  1. LOAD    - sample text about climate change
  2. CHUNK   - RecursiveCharacterTextSplitter (300 chars, 30 overlap)
  3. EMBED   - NVIDIAEmbeddings (nvidia/nv-embedqa-e5-v5)
  4. STORE   - FAISS vector store (in-memory)
  5. RETRIEVE- similarity_search_with_relevance_scores(query, k=2)

Key concepts:

  - Chunk size balances context vs precision. Too big = diluted
    relevance; too small = fragmented meaning.
  - Overlap preserves meaning across chunk boundaries.
  - Embeddings convert text to dense vectors; semantic similarity
    is measured by vector distance.
  - FAISS is fast for in-memory similarity search. For persistence,
    use vectorstore.save_local() / load_local().

This is the retrieval half of RAG. The generation half would feed
these chunks into an LLM prompt.
""")
