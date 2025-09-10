import os
import requests
import streamlit as st

# Load API key from Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"

def call_llm(prompt, debug: bool = False):
    """
    Call Groq's LLaMA3 API with a given prompt.
    Returns the response text or error message.
    """

    if not GROQ_API_KEY:
        return "❌ No API key found. Please configure GROQ_API_KEY in .streamlit/secrets.toml"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful AI document assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 512  # ✅ required to avoid 400 errors
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()

        if debug:
            st.sidebar.write("🔍 Debug Response:", data)

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        error_text = response.text if "response" in locals() else str(e)
        return f"❌ API error: {e}\n\n{error_text}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"
