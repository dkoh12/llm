import requests
import json
from pprint import pprint
from openai import OpenAI

class LMStudioAPI:
    def __init__(self, server: str = "http://localhost:1234", api_key: str = "", openai_api: bool = False, session_history: list = None):
        """
        Initialize the LMStudioAPI client.

        Args:
            server (str): The base URL of the LM Studio server.
            api_key (str): API key for authentication (not used by default).
            openai_api (bool): Whether to use OpenAI-compatible endpoints.
            session_history (list): Initial conversation history (list of dicts).
        """
        self.server = server
        self.api_key = api_key
        self.openai_api = openai_api
        self.client = OpenAI(
            base_url=server + "/v1",
            api_key="lm-studio" # Default API key for LM Studio's OpenAI compatible endpoint
        )
        self.session_history = session_history if session_history is not None else [{"role": "system", "content": "You are a helpful assistant."}]

    def get_lm_studio_models(self) -> None:
        """
        Fetch and print the list of available model IDs from the LM Studio server.
        Uses OpenAI or native endpoint depending on the openai_api flag.
        """
        if self.openai_api:
            api_endpoint = self.server + "/v1/models"
        else:
            api_endpoint = self.server + "/api/v0/models"

        try:
            response = requests.get(api_endpoint)
            if response.status_code == 200:
                data = response.json()
                pprint([model["id"] for model in data.get("data", [])])
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error: {e}")

    def get_single_model(self, model: str) -> None:
        """
        Fetch and print details for a single model.

        Args:
            model (str): The model ID to fetch details for.
        """
        if self.openai_api:
            api_endpoint = self.server + f"/v1/models/{model}"
        else:
            api_endpoint = self.server + f"/api/v0/models/{model}"

        try:
            response = requests.get(api_endpoint)
            if response.status_code == 200:
                pprint(response.json())
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error: {e}")

    def call_chat_completions(self, prompt: str, model: str = "llama-3.2-3b-instruct") -> None:
        """
        Send a chat completion request to the LM Studio server and print the response.
        Appends the user's prompt and AI's response to self.session_history.

        Args:
            prompt (str): The user's prompt.
            model (str): Model ID to use for chat completion.
        """
        # Append the user's prompt to the conversation history
        self.session_history.append({"role": "user", "content": prompt})

        if self.openai_api:
            api_endpoint = self.server + "/v1/chat/completions"
        else:
            api_endpoint = self.server + "/api/v0/chat/completions"

        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": model,
            "messages": self.session_history,
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }

        try:
            response = requests.post(api_endpoint, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                print(answer)
                # Append the AI's response to the conversation history
                self.session_history.append({"role": "assistant", "content": answer})
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error: {e}")

    def completions(self, prompt: str, model: str = "llama-3.2-3b-instruct") -> None:
        """
        Send a text completion request to the LM Studio server and print the response.
        Appends the user's prompt and AI's response to self.session_history.

        Args:
            prompt (str): The prompt string to complete.
            model (str): Model ID to use for completion.
        """
        # Append the user's prompt to the conversation history
        self.session_history.append({"role": "user", "content": prompt})

        if self.openai_api:
            api_endpoint = self.server + "/v1/completions"
        else:
            api_endpoint = self.server + "/api/v0/completions"

        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": model,
            "prompt": prompt, # For standard completions, the full history might not be sent as "prompt"
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False,
            "stop": "\n"
        }

        try:
            response = requests.post(api_endpoint, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["text"]
                complete_sentence = prompt + answer # Or just answer, depending on desired output
                print(complete_sentence)
                # Append the AI's response to the conversation history
                self.session_history.append({"role": "assistant", "content": answer})
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error: {e}")

    def embeddings(self, input_text: str = "Some text to embed", model: str = "text-embedding-nomic-embed-text-v1.5") -> None:
        """
        Request embeddings for the given input text and print the embedding vector.

        Args:
            input_text (str): The text to embed.
            model (str): Model ID to use for embeddings.
        """
        if self.openai_api:
            api_endpoint = self.server + "/v1/embeddings"
        else:
            api_endpoint = self.server + "/api/v0/embeddings"
        
        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": model,
            "input": input_text
        }

        try:
            response = requests.post(api_endpoint, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                answer = data["data"][0]["embedding"]
                print(answer)
            elif response.status_code == 404:
                data = response.json()
                error_message = data.get("error", {}).get("message", "Not found")
                print(error_message)
            else:
                print("Error: ", response.status_code)
                print(response.text)
        except Exception as e:
            print(f"Error: {e}")

    def get_lm_studio_models_openai(self) -> None:
        """
        Fetch and print the list of models using the OpenAI Python client.
        """
        try:
            models= self.client.models.list()
            pprint(models.data)
        except Exception as e:
            print(f"Error: {e}")

    def get_chat_completion_openai(self, prompt: str, model: str = "llama-3.2-3b-instruct") -> None:
        """
        Run a multi-turn chat completion using the OpenAI-compatible API and print the response.
        Appends the user's prompt and AI's response to self.session_history.

        Args:
            prompt (str): The user's prompt.
            model (str): The model ID to use for chat.
        """
        # Ensure self.session_history exists (already handled by __init__)
        # Append the user's prompt to the conversation history
        self.session_history.append({"role": "user", "content": prompt})
        try:
            chat_completion = self.client.chat.completions.create(
                model=model,
                messages=self.session_history,
                temperature=0.7,
            )
            ai_message = chat_completion.choices[0].message.content
            print(ai_message)
            # Append the AI's response to the conversation history
            self.session_history.append({"role": "assistant", "content": ai_message})
        except Exception as e:
            print(f"Error: {e}")


if __name__=="__main__":
    lm_studio_api = LMStudioAPI(openai_api=False)

    # lm_studio_api.get_lm_studio_models()
    # lm_studio_api.get_single_model(model="llama-3.2-3b-instruct") # Replace with an actual model ID if needed

    # lm_studio_api.call_chat_completions(prompt="Hello there!")
    # print(f"Session History: {lm_studio_api.session_history}")

    # lm_studio_api.completions(prompt="Once upon a time, in a land far away")
    # print(f"Session History after completion: {lm_studio_api.session_history}")
    
    # lm_studio_api_openai = LMStudioAPI(openai_api=True)
    # lm_studio_api_openai.get_lm_studio_models_openai()
    # lm_studio_api_openai.get_chat_completion_openai(prompt="Tell me a story about a robot.")
    # print(f"OpenAI Session History: {lm_studio_api_openai.session_history}")

    # lm_studio_api.embeddings(input_text="This is a test sentence for embeddings.")

