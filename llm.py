import os
import requests
import streamlit as st

# ✅ Fix: use [] instead of () for st.secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ GROQ_API_KEY missing in Streamlit secrets. Please set it in .streamlit/secrets.toml")
    GROQ_API_KEY = None
else:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"

def call_llm(prompt):
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
        "temperature": 0.2
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"❌ API error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"
