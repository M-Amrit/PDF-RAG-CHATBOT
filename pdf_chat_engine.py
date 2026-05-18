"""
pdf_chat_engine.py
==================
A production-grade Retrieval-Augmented Generation (RAG) pipeline for
conversational querying over PDF documents.

Architecture
------------
  PDFChatEngine
  ├── Ingestion   – PDF parsing + recursive chunking
  ├── Retrieval   – FAISS vector store + similarity search
  └── Generation  – GPT-4o with a grounded-answering system prompt

Author : Staff AI Engineer
Python : 3.10+
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grounded-answering system prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a precise document assistant. Your ONLY knowledge
source is the context passages retrieved from the user's PDF.

Rules you must never break:
1. Answer exclusively from the provided context — never from prior training
   knowledge or general world knowledge.
2. If the context does not contain enough information to answer confidently,
   respond with exactly:
   "I could not find an answer to that question in the provided document."
3. Do not speculate, infer beyond what is written, or fabricate citations.
4. Keep answers concise and factual; quote relevant sentences when helpful.

Context:
---------
{context}
---------

Question: {question}

Answer:"""

_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
class PDFChatEngine:
    """End-to-end RAG pipeline for PDF question-answering.

    Usage::

        engine = PDFChatEngine()
        engine.ingest("report.pdf")
        answer = engine.query("What is the revenue for Q3?")
        print(answer)

    Args:
        model_name: OpenAI chat model identifier (default ``gpt-4o``).
        embedding_model: OpenAI embedding model (default
            ``text-embedding-3-small``).
        chunk_size: Target character length per chunk.  1 000 chars ≈ 250
            tokens, comfortably within embedding limits while preserving
            paragraph-level context.
        chunk_overlap: Characters shared between adjacent chunks.  200-char
            overlap ensures that sentences split at a boundary still appear
            intact in at least one chunk, preventing context fragmentation.
        retrieval_k: Number of chunks returned per query.  k=4 typically
            gives enough context (~4 000 chars) without saturating the
            prompt window.
        temperature: LLM sampling temperature.  0.0 maximises determinism
            and factual grounding; raise slightly (≤ 0.3) for more fluent
            phrasing if needed.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        retrieval_k: int = 4,
        temperature: float = 0.0,
    ) -> None:
        self._validate_env()

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k

        # Text splitter shared across all ingestion calls
        self._splitter = RecursiveCharacterTextSplitter(
            # RecursiveCharacterTextSplitter is preferred over fixed-size
            # splitting because it tries sentence/paragraph boundaries first,
            # only falling back to hard character cuts when necessary.
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        self._embeddings = OpenAIEmbeddings(model=embedding_model)
        self._llm = ChatOpenAI(model=model_name, temperature=temperature)

        self._vectorstore: Optional[FAISS] = None
        self._chain = None  # LCEL runnable: retriever | prompt | llm | parser

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, pdf_path: str | Path) -> None:
        """Parse a PDF, chunk it, embed the chunks, and build the FAISS index.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Raises:
            FileNotFoundError: If ``pdf_path`` does not exist.
            RuntimeError: If PDF parsing or embedding fails.
        """
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Ingesting PDF: %s", pdf_path.name)

        raw_pages = self._load_pdf(pdf_path)
        chunks = self._chunk_documents(raw_pages)

        logger.info(
            "Parsed %d pages → %d chunks (size=%d, overlap=%d)",
            len(raw_pages),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )

        self._build_vectorstore(chunks)
        self._build_chain()
        logger.info("Index ready. Engine is accepting queries.")

    def query(self, question: str) -> str:
        """Retrieve relevant chunks and generate a grounded answer.

        Args:
            question: Natural-language question about the ingested document.

        Returns:
            A factual answer derived solely from document context, or a
            polite "not found" message if the answer is absent.

        Raises:
            RuntimeError: If :meth:`ingest` has not been called yet.
        """
        if self._chain is None:
            raise RuntimeError(
                "No document has been ingested. Call engine.ingest(path) first."
            )

        if not question.strip():
            return "Please provide a non-empty question."

        logger.debug("Query: %s", question)

        try:
            # LCEL chains return the final parsed value directly
            return self._chain.invoke(question).strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise RuntimeError(f"Generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_env() -> None:
        """Ensure required API keys are present before any network calls."""
        load_dotenv()  # Honour a local .env file if present
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set.  "
                "Export it or add it to a .env file in the project root."
            )

    def _load_pdf(self, pdf_path: Path) -> list[Document]:
        """Load and parse a PDF into per-page LangChain Document objects.

        Args:
            pdf_path: Resolved path to the PDF.

        Returns:
            List of :class:`~langchain_core.documents.Document` instances,
            one per page.

        Raises:
            RuntimeError: On any I/O or parsing error.
        """
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
        except Exception as exc:
            raise RuntimeError(f"Failed to parse '{pdf_path.name}': {exc}") from exc

        if not pages:
            raise RuntimeError(f"'{pdf_path.name}' appears to be empty or unreadable.")

        return pages

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split raw pages into overlapping chunks for embedding.

        Args:
            documents: Per-page Document objects from the PDF loader.

        Returns:
            Flat list of chunked Document objects with inherited metadata.
        """
        return self._splitter.split_documents(documents)

    def _build_vectorstore(self, chunks: list[Document]) -> None:
        """Embed chunks and build an in-memory FAISS index.

        FAISS is chosen over alternatives (Chroma, Pinecone) here because:
        - Zero infra overhead — suitable for single-document workloads.
        - Sub-millisecond query latency for corpora up to ~100k chunks.
        - The index can be serialised to disk via ``FAISS.save_local()``
          when persistence is needed.

        Args:
            chunks: Chunked Document objects to embed and index.

        Raises:
            RuntimeError: On embedding API failure.
        """
        try:
            self._vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self._embeddings,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to build vector index: {exc}") from exc

    def _build_chain(self) -> None:
        """Wire the retriever, prompt, LLM, and parser into an LCEL chain.

        The pipeline is built with LangChain Expression Language (LCEL), the
        current idiomatic approach (``RetrievalQA`` is deprecated as of
        LangChain 0.2).  Data flows as:

            question (str)
              → retriever          – fetches top-k chunks from FAISS
              → prompt             – injects chunks + question into the template
              → llm                – calls GPT-4o
              → StrOutputParser    – extracts the plain-text reply

        The ``stuff`` strategy (concatenating all chunks into one prompt) is
        appropriate here because k=4 chunks × 1 000 chars ≈ 1 000 tokens,
        comfortably within GPT-4o's 128k context window.
        """
        retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.retrieval_k},
        )

        def _format_docs(docs: list[Document]) -> str:
            """Flatten retrieved chunks into a single context string."""
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        # LCEL pipe: each step's output becomes the next step's input.
        self._chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | _PROMPT_TEMPLATE
            | self._llm
            | StrOutputParser()
        )


# ---------------------------------------------------------------------------
# Interactive CLI loop (optional standalone usage)
# ---------------------------------------------------------------------------
def run_interactive_session(engine: PDFChatEngine) -> None:
    """Start a read-eval-print loop for conversational PDF querying.

    Args:
        engine: A fully ingested :class:`PDFChatEngine` instance.
    """
    print("\n" + "=" * 60)
    print("  PDF Chat Engine — type 'exit' or 'quit' to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        answer = engine.query(question)
        print(f"\nAssistant: {answer}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ── Instantiate the engine ──────────────────────────────────────────────
    engine = PDFChatEngine(
        model_name="gpt-4o",
        embedding_model="text-embedding-3-small",
        chunk_size=1000,
        chunk_overlap=200,
        retrieval_k=4,
        temperature=0.0,
    )

    # ── Ingest the target PDF ───────────────────────────────────────────────
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    try:
        engine.ingest(pdf_file)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("Ingestion failed: %s", e)
        sys.exit(1)

    # ── Single programmatic query (useful for CI / smoke tests) ────────────
    test_question = "What is the main topic of this document?"
    answer = engine.query(test_question)
    print(f"\nQ: {test_question}\nA: {answer}\n")

    # ── Drop into the interactive chat loop ────────────────────────────────
    run_interactive_session(engine)