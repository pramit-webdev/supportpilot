import requests
import streamlit as st

# Get your Groq API key from secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"

def stream_llm_response(prompt):
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
        "stream": True,
        "temperature": 0.2
    }

    try:
        with requests.post(GROQ_API_URL, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    if line.decode().strip() == "data: [DONE]":
                        break
                    try:
                        data = line.decode().replace("data: ", "")
                        token = eval(data)["choices"][0]["delta"].get("content", "")
                        if token:
                            yield token
                    except Exception:
                        continue
    except requests.exceptions.RequestException as e:
        yield f"\n⚠️ API request failed: {e}"
