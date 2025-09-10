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
        "temperature": 0.2,
        "max_tokens": 512  # ✅ prevent 400 error
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"❌ API error: {e} | {response.text if 'response' in locals() else ''}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"
