# PDF RAG Chatbot

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot that lets you have a conversation with any PDF document. Ask questions in plain English and get precise, grounded answers pulled directly from your document — powered by **OpenAI GPT-4o** and **LangChain**.

> No hallucinations. If the answer is not in the PDF, the bot says so.

---

## Features

- Chat with any PDF using natural language
- Semantic search powered by FAISS vector store
- GPT-4o for accurate, context-aware answers
- Grounded answering — strictly no hallucinations
- Fast in-memory vector indexing with no database setup required
- Simple CLI interface — just run and type

---

## How It Works

```
Your Question
      |
      v
FAISS Vector Store  -->  Top 4 relevant chunks from the PDF
      |
      v
GPT-4o (LLM)  -->  Answer grounded in those chunks only
```

**Ingestion** — The PDF is parsed page by page, split into overlapping chunks, converted into vector embeddings, and stored in FAISS.

**Retrieval** — Your question is embedded and the 4 most semantically similar chunks are fetched from FAISS.

**Generation** — GPT-4o reads only those chunks and answers your question strictly from that context.

---

## Project Structure

```
pdf-rag-chatbot/
├── pdf_chat_engine.py   # Full RAG pipeline (ingestion, retrieval, generation)
├── requirements.txt     # All Python dependencies
├── .env                 # Your OpenAI API key (you create this — never commit it)
├── .gitignore           # Keeps your API key and PDFs out of Git
└── README.md            # This file
```

---

## Tech Stack

- **Language Model** — OpenAI GPT-4o
- **Embeddings** — OpenAI text-embedding-3-small
- **Vector Store** — FAISS (local, in-memory)
- **PDF Parser** — PyPDF via LangChain Community
- **Chain Orchestration** — LangChain LCEL
- **Environment Config** — python-dotenv

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- An OpenAI account with billing enabled — [platform.openai.com/billing](https://platform.openai.com/billing)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 — Set up your API key

Create a file named `.env` in the project root folder and add your OpenAI API key inside it:

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

Get your key from: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

> **Warning:** Never share this file or push it to GitHub. It is already excluded via `.gitignore`.

---

### Step 4 — Run the chatbot

```bash
python pdf_chat_engine.py sample.pdf
```

Replace `sample.pdf` with the path to your own PDF file. You can also pass a full path:

```bash
python pdf_chat_engine.py "C:\Users\yourname\Documents\report.pdf"
```

---

## Example Session

```
15:42:01  INFO   Ingesting PDF: sample.pdf
15:42:03  INFO   Parsed 99 pages - 312 chunks (size=1000, overlap=200)
15:42:09  INFO   Index ready. Engine is accepting queries.

Q: What is the main topic of this document?
A: The document covers ...

============================================================
  PDF Chat Engine  --  type 'exit' or 'quit' to stop
============================================================

You: What are the key findings in chapter 3?
Assistant: The key findings in chapter 3 are ...

You: Who wrote this report?
Assistant: I could not find an answer to that question in the provided document.

You: exit
Goodbye.
```

---

## Configuration

You can adjust these parameters inside `pdf_chat_engine.py` when instantiating the engine:

- **model_name** *(default: gpt-4o)* — OpenAI model used to generate answers
- **embedding_model** *(default: text-embedding-3-small)* — Model used to convert text into vectors
- **chunk_size** *(default: 1000)* — Number of characters per chunk
- **chunk_overlap** *(default: 200)* — Characters shared between adjacent chunks to preserve context
- **retrieval_k** *(default: 4)* — Number of chunks retrieved per question
- **temperature** *(default: 0.0)* — 0 = deterministic and factual, higher = more creative

---

## Common Errors and Fixes

**OPENAI_API_KEY is not set**
Your `.env` file is missing or incorrectly named. Create a file named exactly `.env` in the same folder as the script.

**PDF not found**
The path to your PDF is wrong. Pass the full path to the file when running the script.

**429 insufficient_quota**
Your OpenAI account has no credits. Add billing at [platform.openai.com/billing](https://platform.openai.com/billing).

**ModuleNotFoundError**
Dependencies are not installed. Run `pip install -r requirements.txt`.

---

## Cost Estimate

A typical session chatting with a 100-page PDF costs roughly $0.01 to $0.05 depending on how many questions you ask. A $5 credit top-up will last a very long time for personal use.

---

## Security Notes

- Your `.env` file is listed in `.gitignore` and will never be pushed to GitHub
- Never paste your API key into any chat, email, or public forum
- If your key is ever exposed, revoke it immediately at [platform.openai.com/api-keys](https://platform.openai.com/api-keys) and generate a new one

---

## License

This project is open source and available under the [MIT License](LICENSE).
