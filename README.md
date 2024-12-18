# RAG System with PDF Ingestion

An advanced Retrieval-Augmented Generation (RAG) system that combines document processing, semantic search, and large language models to provide intelligent answers to user queries. The system features both standard and agentic RAG approaches, along with PDF document ingestion capabilities.

## Table of Contents
1. [System Overview](#system-overview)
2. [Key Features](#key-features)
3. [Technical Architecture](#technical-architecture)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
6. [Components Deep Dive](#components-deep-dive)
7. [Educational Insights](#educational-insights)
8. [Key Takeaways](#key-takeaways)
9. [Advanced Concepts](#advanced-concepts)
10. [Troubleshooting](#troubleshooting)

## System Overview

This system implements a sophisticated RAG pipeline that:
- Processes and indexes documents using semantic embeddings
- Supports dynamic PDF ingestion
- Provides two RAG approaches: standard and agentic
- Uses Google's Gemini model for generation
- Implements efficient vector similarity search using FAISS

## Key Features

1. **Dual RAG Implementation**
   - Standard RAG: Direct retrieval and generation
   - Agentic RAG: Multi-step reasoning with tool usage

2. **Document Processing**
   - PDF document ingestion
   - Recursive text splitting with overlap
   - Duplicate content detection
   - Semantic chunking

3. **Vector Search**
   - FAISS vector database integration
   - Cosine similarity search
   - Efficient embedding caching

4. **LLM Integration**
   - Google Gemini model integration
   - Streamlit web interface
   - Configurable generation parameters

## Technical Architecture

```
                                    ┌─────────────────┐
                                    │   PDF Upload    │
                                    └────────┬────────┘
                                            │
                                    ┌───────▼────────┐
┌─────────────────┐                │   Document     │
│  Knowledge Base ├───────────────►│   Processing   │
└─────────────────┘                └───────┬────────┘
                                          │
                                   ┌──────▼─────────┐
                                   │    Vector DB   │
                                   │    (FAISS)     │
                                   └──────┬─────────┘
                                         │
                        ┌────────────────┴───────────────┐
                        │                                │
                 ┌──────▼──────┐                 ┌──────▼──────┐
                 │  Standard   │                 │   Agentic   │
                 │    RAG      │                 │    RAG      │
                 └──────┬──────┘                 └──────┬──────┘
                        │                               │
                        └───────────────┬───────────────┘
                                       │
                                ┌──────▼──────┐
                                │   Gemini    │
                                │    LLM      │
                                └──────┬──────┘
                                       │
                                ┌──────▼──────┐
                                │  Response   │
                                └─────────────┘
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GOOGLE_API_KEY='your-gemini-api-key'
```

## Usage Guide

1. Start the application:
```bash
streamlit run app.py
```

2. Upload PDFs through the web interface
3. Enter questions in the text input
4. Compare answers from both RAG approaches

## Components Deep Dive

### Document Processing Pipeline
```python
def process_documents(source_docs):
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True
    )
```
- Uses GTE-small tokenizer for consistent chunking
- Implements recursive splitting with overlap
- Maintains document metadata

### Vector Database Integration
```python
def initialize_vectordb(_docs_processed):
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    vectordb = FAISS.from_documents(
        documents=_docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
```
- Uses FAISS for efficient similarity search
- Implements cosine distance strategy
- Caches embeddings for performance

## Educational Insights

### RAG Architecture Patterns

1. **Standard RAG Pattern**
   - Single-step retrieval and generation
   - Direct context injection
   - Suitable for straightforward queries

2. **Agentic RAG Pattern**
   - Multi-step reasoning
   - Tool-based information gathering
   - Better for complex queries requiring synthesis

### Vector Search Concepts

1. **Embedding Space**
   - Documents are converted to high-dimensional vectors
   - Similarity is measured by vector distance
   - Cosine similarity provides normalized comparison

2. **Chunking Strategy**
   - Balance between context size and relevance
   - Overlap prevents context loss at boundaries
   - Recursive splitting respects semantic boundaries

## Key Takeaways

1. **System Design**
   - Modular architecture enables easy extensions
   - Caching strategies improve performance
   - Dual RAG approaches provide comparison opportunities

2. **Performance Optimization**
   - Document deduplication reduces index size
   - Embedding caching speeds up retrieval
   - Chunking parameters affect result quality

3. **User Experience**
   - Interactive web interface with Streamlit
   - Real-time PDF ingestion
   - Comparative answer generation

## Advanced Concepts

### Custom Retriever Tool
```python
class RetrieverTool(Tool):
    def forward(self, query: str) -> str:
        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )
```
- Implements tool-based retrieval interface
- Configurable number of retrieved documents
- Formats results for agent consumption

### Gemini Integration
```python
class GeminiEngine:
    def __call__(self, messages, stop_sequences=[]):
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        response = self.model.generate_content(prompt)
```
- Handles message formatting
- Configurable generation parameters
- Stop sequence support

## Troubleshooting

Common issues and solutions:

1. **Memory Usage**
   - Use chunking parameters appropriate for available RAM
   - Implement batch processing for large documents
   - Clear cache periodically

2. **Query Performance**
   - Optimize number of retrieved documents
   - Tune chunking parameters
   - Use appropriate embedding model size

3. **Result Quality**
   - Adjust chunk overlap for better context
   - Try different embedding models
   - Tune the number of agent iterations

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any enhancements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
