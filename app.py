import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer

st.set_page_config(page_title="SupportPilot", layout="wide")
st.title("🤖 SupportPilot")

# --- Sidebar: Upload PDFs ---
st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    handle_upload(uploaded_files)
    st.sidebar.success("✅ Documents uploaded and indexed.")

# --- Sidebar: Reset Button ---
if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.sidebar.warning("All documents and index have been cleared!")

# --- Chat History State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Ask me anything related to your uploaded documents."}
    ]

# --- Display chat messages like ChatGPT ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("ai"):
            st.markdown(msg["content"])

# --- User Input ---
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process + Get AI response
    with st.chat_message("ai"):
        with st.spinner("SupportPilot is thinking..."):
            response = get_answer(user_input)
            st.markdown(response)

    # Save response
    st.session_state.messages.append({"role": "ai", "content": response})
