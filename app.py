import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer

st.set_page_config(page_title="SupportPilot", layout="wide")
st.title("🤖 SupportPilot – AI Customer Support Assistant")

# --- Sidebar Upload ---
st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    handle_upload(uploaded_files)
    st.sidebar.success("✅ Documents uploaded and indexed.")

# --- Sidebar Reset ---
if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.sidebar.warning("🧹 All documents and index cleared.")

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Ask me anything from your uploaded documents."}
    ]

# --- Display Chat Messages (ChatGPT style) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream AI response
    with st.chat_message("ai"):
        response_stream = get_answer(user_input)
        full_response = st.write_stream(response_stream)

    # Add AI response to history
    st.session_state.messages.append({"role": "ai", "content": full_response})
