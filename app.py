import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer
import os

st.set_page_config(page_title="SupportPilot", layout="wide")
st.title("🤖 SupportPilot – AI Customer Support Assistant")

# Debug mode toggle
debug_mode = st.sidebar.checkbox("🛠 Show debug logs")
st.session_state.debug = debug_mode

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Upload your PDFs and ask me anything!"}
    ]

# Upload PDFs one-by-one
st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with st.spinner(f"🔄 Processing `{file.name}`..."):
            success = handle_upload([file])
        if success:
            st.sidebar.success(f"✅ `{file.name}` indexed.")
        else:
            st.sidebar.warning(f"⚠️ `{file.name}` already indexed or contains no text.")

# Reset system
if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.sidebar.warning("🧹 All documents and index cleared.")
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Upload your PDFs and ask me anything!"}
    ]
    st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Check if FAISS index exists
if not os.path.exists("data/faiss_index/support_index.faiss"):
    st.info("📄 Please upload at least one support PDF to start chatting.")
    st.stop()

# Chat input
user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("ai"):
        with st.spinner("💬 Generating response..."):
            response, context_chunks = get_answer(user_input)
            st.markdown(response)

            if context_chunks:
                with st.expander("📄 Sources used"):
                    for chunk in context_chunks:
                        st.markdown(f"```text\n{chunk}\n```")

    st.session_state.messages.append({"role": "ai", "content": response})
