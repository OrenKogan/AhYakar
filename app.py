import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Global conversation history (acts as memory while the server runs)
chat_history = [
    {
        "role": "system", 
        "content": (
            "You are a specialized Medical Assistant. "
            "1. If a symptom sounds serious, advise seeking emergency care immediately. "
            "2. Suggest ONLY Over-The-Counter (OTC) medications. "
            "3. State clearly that you are an AI and not a doctor."
        )
    }
]

@app.route('/')
def home():
    """Serves the web GUI."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles the chat logic and AI generation."""
    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # Add user message to history
    chat_history.append({"role": "user", "content": user_message})

    try:
        # Call the AI model
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash", 
            messages=chat_history,
            extra_headers={
                "HTTP-Referer": "http://localhost:5000",
                "X-OpenRouter-Title": "AhYakar Medical Bot",
            },
            temperature=0.2 # Low temperature for factual medical advice
        )

        ai_reply = response.choices[0].message.content
        
        # Add AI reply to history
        chat_history.append({"role": "assistant", "content": ai_reply})

        return jsonify({"reply": ai_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)