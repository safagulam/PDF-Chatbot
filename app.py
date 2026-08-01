import streamlit as st
from rag import build_vectorstore, ask_question

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Chatbot")
st.markdown("Ask questions about your PDF using AI.")

# -------------------------------------------------
# Session State
# -------------------------------------------------

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

with st.sidebar:

    st.header("📄 Upload PDF")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_pdf is not None:

        if st.button("Process PDF"):

            with st.spinner("Processing PDF..."):

                vector_db, chunks = build_vectorstore(uploaded_pdf)

                st.session_state.vector_db = vector_db
                st.session_state.chunks = chunks

            st.success("✅ PDF processed successfully!")

# -------------------------------------------------
# Display Chat History
# -------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------
# Chat Input
# -------------------------------------------------

user_question = st.chat_input("Ask a question about the PDF...")

if user_question:

    if st.session_state.vector_db is None:

        st.warning("Please upload and process a PDF first.")
        st.stop()

    # -----------------------
    # Display User Message
    # -----------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # -----------------------
    # Generate Answer
    # -----------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, docs = ask_question(
            st.session_state.vector_db,
            st.session_state.chunks,
            user_question
        )

        st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # -----------------------
        # Show Source Chunks
        # -----------------------

        with st.expander("📄 View Retrieved Chunks"):

            for i, doc in enumerate(docs, start=1):

                page = doc.metadata.get("page", "Unknown")

                st.markdown(f"### Chunk {i} (Page {page})")
                st.write(doc.page_content)
                st.divider()