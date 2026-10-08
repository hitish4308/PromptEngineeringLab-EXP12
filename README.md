# Prompt Engineering Lab - Experiment 12

## RAG Data Indexing Pipeline

Loads sample text, chunks it, embeds it, stores it in a FAISS vector
store, and retrieves the most relevant chunks for a query. This is the
retrieval half of a Retrieval-Augmented Generation (RAG) pipeline.

## Pipeline Stages

| Stage | Tool | Detail |
|-------|------|--------|
| Load | Python string | Sample paragraph about climate change |
| Chunk | RecursiveCharacterTextSplitter | 300 chars, 30 overlap |
| Embed | NVIDIAEmbeddings | nvidia/nemotron-3-embed-1b |
| Store | FAISS | In-memory vector store |
| Retrieve | similarity_search_with_relevance_scores | Top 2 chunks + scores |

## Model and Parameters

- Embedding model: nvidia/nemotron-3-embed-1b (NVIDIA NIM)
- Chunk size: 300 chars
- Overlap: 30 chars
- Query: "What are the effects of climate change?"
- Top-k retrieval: 2

## Why NVIDIAEmbeddings, not OpenAIEmbeddings

The lab manual suggests OpenAI's text-embedding-ada-002. Since this
environment is configured for NVIDIA NIM, we use the official
NVIDIAEmbeddings integration. It works with the same NVIDIA_API_KEY
already in .env and supports both query and passage embedding modes.

## Why nvidia/nemotron-3-embed-1b

The original candidate, nvidia/nv-embedqa-e5-v5, was retired by NVIDIA
on 2026-08-25 and now returns HTTP 410 Gone. The current live model
for RAG embeddings is nvidia/nemotron-3-embed-1b.

## Setup

Reuse the environment from Experiment 1:

    Copy-Item ..\EXP_1\.env .
    python -m pip install -r requirements.txt

Requirements now include:

    langchain>=0.3.0
    langchain-community>=0.3.0
    langchain-core>=0.3.0
    langchain-nvidia-ai-endpoints>=0.3.0
    faiss-cpu>=1.8.0

## Run

    python rag_indexing.py

## Expected Output

- Number of chunks created (typically 3-4 for the sample text)
- Top 2 most relevant chunks with similarity scores
- A summary of the RAG indexing concepts

Example:

    Step 1: Chunking text...
      Number of chunks: 4

    Step 3: Building FAISS vector store...
      Vector store created with 4 chunks.

    QUERY: What are the effects of climate change?

    Top 2 chunks:

      --- Result 1 ---
      Score: 0.7523
      Chunk: Rising global temperatures cause sea levels to rise...

      --- Result 2 ---
      Score: 0.6891
      Chunk: More frequent and intense storms, droughts...

## Key Findings

1. **Chunking** - 300/30 splits text into readable semantic units
   without cutting sentences mid-thought. Overlap preserves meaning
   across chunk boundaries.

2. **Embeddings** - dense vectors capture meaning; similar ideas
   cluster together in vector space.

3. **FAISS** - fast similarity search. For persistence, use
   vectorstore.save_local() / load_local().

4. **Retrieval quality** depends on chunk size, embedding model
   quality, and query phrasing.

## Troubleshooting

### Error code: 410 - Gone

The embedding model was retired by NVIDIA. Swap to a currently-served
model. Check the live list:

    curl -s -H "Authorization: Bearer $env:NVIDIA_API_KEY" https://integrate.api.nvidia.com/v1/models

Look for entries containing "embed" and use the model ID shown.

### Error code: 403 - Forbidden

NVIDIAEmbeddings needs NVIDIA_API_KEY in the environment. Ensure
.env is present and load_dotenv() runs before the embeddings object
is created.

### faiss-cpu install failure on Windows

If pip cannot build FAISS, tell the lab instructor to swap to a
NumPy-based in-memory vector store. The rest of the pipeline remains
the same.

### No chunks or only 1 chunk

The sample text is too short. Extend it to at least 400 characters or
reduce the chunk size.

## Next Step (RAG Generation)

This is only the retrieval half. A full RAG pipeline would:

1. Retrieve top-k chunks (this experiment)
2. Inject them into an LLM prompt as context
3. Generate an answer grounded in the retrieved text

## Security

- Never commit .env
- Never paste API keys in chat, logs, or screenshots

## License

For educational / lab use only.
