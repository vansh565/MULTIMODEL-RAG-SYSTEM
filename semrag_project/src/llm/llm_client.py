# src/llm/llm_client.py
import requests
import json

def generate(prompt, model="mistral"):
    """Generate response using Ollama with Mistral model"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated")
        else:
            print(f"Ollama API error: {response.status_code}")
            return "Error: Unable to connect to Ollama. Please make sure Ollama is running."
    
    except requests.exceptions.ConnectionError:
        print("Connection error: Make sure Ollama is running with 'ollama serve'")
        return "Error: Cannot connect to Ollama. Please run 'ollama serve' in terminal."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Error: {str(e)}"

def generate_stream(prompt, model="mistral"):
    """Generate streaming response using Ollama"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    yield data.get("response", "")
                except:
                    pass
    except Exception as e:
        print(f"Streaming error: {e}")
        yield f"Error: {str(e)}"

def check_mistral_available():
    """Check if Mistral model is available in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                if "mistral" in model.get("name", "").lower():
                    return True
        return False
    except:
        return False