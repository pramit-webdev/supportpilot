import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer

st.set_page_config(page_title="SupportPilot", layout="wide")
st.title("🤖 SupportPilot - AI Customer Support Agent")

# Sidebar Upload
st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    handle_upload(uploaded_files)
    st.sidebar.success("✅ Documents uploaded and indexed.")

# Sidebar Reset Button
if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.sidebar.warning("All documents and index have been cleared!")

# --- Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
query = st.text_input("💬 You:", key="query_input")

# Submit Button
if st.button("Get Answer"):
    if query.strip():
        with st.spinner("🤔 Thinking..."):
            answer = get_answer(query)

        # Save to chat history
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("SupportPilot", answer))
    else:
        st.warning("Please enter a question.")

# Show Chat History
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"🧍 **You**: {msg}")
    else:
        st.markdown(f"🤖 **SupportPilot**: {msg}")
