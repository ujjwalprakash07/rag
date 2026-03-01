# RAG using OpenAI (Python)

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** system
built using Python and OpenAI APIs.

## Features
- Converts text into embeddings using OpenAI
- Retrieves the most relevant content using semantic similarity
- Generates grounded answers using an LLM
- No heavy ML libraries required

## How it works
1. Text data is split into chunks
2. Each chunk is converted into embeddings
3. User query is embedded and compared using cosine similarity
4. Most relevant chunk is used as context for answer generation

## Setup

### Install dependencies
```bash
pip install openai numpy
