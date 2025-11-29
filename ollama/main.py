import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_response(prompt: str, model: str = "llama3.2:1b") -> str:
    """
    Sends a request to the local model and returns a response.
    """
    payload = {
        "model":"llama3.2:1b",
        "prompt": prompt,
        "stream": False  
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        data = response.json()
        return data.get("response", "[No response from model]")
    except Exception as e:
        return f"Error: {e}"

def chat():
    print("Local chatbot launched! Type 'Wow' to finish.")
    while True:
        user_input = input("\nYou: ")

     
        if user_input.strip().lower() == "вуе":
            print("Chat is complete.")
            break

       
        reply = generate_response(user_input)
        print(f"Bot: {reply}")

if __name__ == "__main__":
    chat()
