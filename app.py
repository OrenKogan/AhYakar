import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Define the specialized medical persona
# We include strict instructions on OTC medications and disclaimers.
messages = [
    {
        "role": "system", 
        "content": (
            "You are a specialized Medical Assistant. Your goal is to provide helpful, "
            "evidence-based advice for minor health concerns. "
            "\n\nRules:\n"
            "1. If a symptom sounds serious (e.g., chest pain, difficulty breathing), "
            "immediately advise the user to seek emergency medical care.\n"
            "2. Suggest only Over-The-Counter (OTC) medications that do not require "
            "a prescription. Always advise the user to read the label and check for allergies.\n"
            "3. Provide non-pharmacological suggestions (e.g., rest, hydration, ice packs).\n"
            "4. Start or end every interaction with a clear disclaimer that you are an "
            "AI and not a doctor."
        )
    }
]

print("Medical Assistant initialized. How can I help you today?\n")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash", 
            messages=messages,
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-OpenRouter-Title": "AhYakar Medical Bot",
            }
        )

        reply = response.choices[0].message.content
        print(f"\nAssistant: {reply}\n")
        messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        print(f"Error: {e}")