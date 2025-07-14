import os
import requests
import streamlit as st

# --- API Settings ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"

def call_llm(prompt: str) -> str:
    """Calls Groq's LLaMA 3 model to get a single response for the given prompt."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        
        if st.session_state.get("debug"):
            st.write("[DEBUG] LLM API call successful.")
            st.code(reply[:500])  # preview

        return reply

    except requests.exceptions.RequestException as e:
        error_msg = f"❌ API request failed: {e}"
        if st.session_state.get("debug"):
            st.error(error_msg)
        return error_msg

    except Exception as e:
        fallback_msg = f"❌ Unexpected error: {e}"
        if st.session_state.get("debug"):
            st.error(fallback_msg)
        return fallback_msg
