from llm import call_llm

def get_answer(query):
    context_chunks = retrieve_context(query)
    if isinstance(context_chunks, str):  # error message
        return context_chunks

    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful AI customer support assistant.

Use the following document context to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

    if st.session_state.get("debug"):
        st.write("[DEBUG] Prompt sent to Groq:")
        st.code(prompt[:800])

    return call_llm(prompt)
