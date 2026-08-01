"""
rag.py

Backend logic for PDF Chatbot (RAG)

Pipeline:
PDF
 ↓
Load PDF
 ↓
Split into chunks
 ↓
Generate embeddings
 ↓
Store in FAISS
 ↓
Retriever
 ↓
Groq LLM
 ↓
Answer
"""

import os
import tempfile

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")


# --------------------------------------------------
# Initialize Embedding Model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Initialize LLM
# --------------------------------------------------

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0
)


# --------------------------------------------------
# Build Vector Store
# --------------------------------------------------

def build_vectorstore(uploaded_file):
    """
    uploaded_file : Streamlit uploaded PDF

    Returns:
        FAISS Vector Store
    """

    # Save uploaded PDF temporarily

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_pdf:

        temp_pdf.write(uploaded_file.read())

        pdf_path = temp_pdf.name


    # Load PDF

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()


    # Split document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)


    # Create FAISS

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_db, chunks

def summarize_document(chunks):
    """
    Hierarchical summarization:
    1. Summarize each chunk.
    2. Merge all summaries into one final summary.
    """

    partial_summaries = []

    for i, chunk in enumerate(chunks):

        prompt = f"""
You are an expert document summarizer.

Summarize ONLY the following section.

Requirements:
- Preserve important facts.
- Mention key technologies, objectives, findings and conclusions.
- Keep it under 150 words.

Document Section:

{chunk.page_content}

Summary:
"""

        response = llm.invoke(prompt)

        partial_summaries.append(response.content)

    combined_summary = "\n\n".join(partial_summaries)

    final_prompt = f"""
You are an expert technical writer.

The following are summaries of different sections of a document.

Create ONE final structured summary.

Include:

1. Document Title (if available)
2. Objective
3. Main Topics
4. Methodology / Workflow
5. Technologies Used
6. Important Findings
7. Conclusion

Section Summaries:

{combined_summary}

Final Summary:
"""

    response = llm.invoke(final_prompt)

    return response.content
# --------------------------------------------------
# Ask Question
# --------------------------------------------------

def ask_question(vector_db, chunks, question):

    summary_keywords = [
        "summary",
        "summarize",
        "summarise",
        "overview",
        "explain the document",
        "summarize the pdf",
    ]

    is_summary = any(
        keyword in question.lower()
        for keyword in summary_keywords
    )

    # -----------------------------
    # SUMMARY MODE
    # -----------------------------
    if is_summary:

        summary = summarize_document(chunks)

        return summary, chunks[:5]

    # -----------------------------
    # QUESTION ANSWERING MODE
    # -----------------------------

    retriever = vector_db.as_retriever(
        search_kwargs={"k":5}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are an expert AI assistant.

Answer ONLY from the provided context.

If the answer cannot be found, reply:

"I couldn't find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, docs