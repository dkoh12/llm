import requests
import json
from pprint import pprint # Keep for printing model lists if desired, or replace with logger
from openai import OpenAI # Removed APIError, APIConnectionError
# Removed requests.exceptions as we'll use general Exception
from logger import get_logger # Assuming your logger file is named logger.py

# Get a logger for this module
logger = get_logger(__name__)

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
        logger.info(f"LMStudioAPI initialized. OpenAI compatible: {self.openai_api}, Server: {self.server}")

    def get_lm_studio_models(self) -> None:
        """
        Fetch and print the list of available model IDs from the LM Studio server.
        Uses OpenAI or native endpoint depending on the openai_api flag.
        """
        if self.openai_api:
            api_endpoint = self.server + "/v1/models"
        else:
            api_endpoint = self.server + "/api/v0/models"
        logger.debug(f"Fetching models from: {api_endpoint}")
        try:
            response = requests.get(api_endpoint, timeout=10)
            response.raise_for_status() # Still good to check for HTTP errors
            data = response.json()
            # pprint([model["id"] for model in data.get("data", [])]) # User-facing output
            print("Available models:") # User-facing output
            for model in data.get("data", []): # User-facing output
                print(f"  - {model.get('id')}") # User-facing output
        except Exception as e:
            logger.exception(f"An error occurred in get_lm_studio_models: {e}")

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
        logger.debug(f"Fetching single model '{model}' from: {api_endpoint}")
        try:
            response = requests.get(api_endpoint, timeout=10)
            response.raise_for_status()
            pprint(response.json()) # User-facing output (pprint is fine for complex dicts)
        except Exception as e:
            logger.exception(f"An error occurred in get_single_model: {e}")

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
        logger.debug(f"Calling chat completions at: {api_endpoint} with model: {model}")
        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": model,
            "messages": self.session_history,
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            print(answer) # User-facing output
            # Append the AI's response to the conversation history
            self.session_history.append({"role": "assistant", "content": answer})
        except Exception as e:
            logger.exception(f"An error occurred in call_chat_completions: {e}")

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
        logger.debug(f"Calling completions at: {api_endpoint} with model: {model}")
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
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["text"]
            complete_sentence = prompt + answer # Or just answer, depending on desired output
            print(complete_sentence) # User-facing output
            # Append the AI's response to the conversation history
            self.session_history.append({"role": "assistant", "content": answer})
        except Exception as e:
            logger.exception(f"An error occurred in completions: {e}")

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
        logger.debug(f"Requesting embeddings from: {api_endpoint} for model: {model}")
        headers = {'Content-Type': 'application/json'}
        payload = {"model": model, "input": input_text}
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            answer = data["data"][0]["embedding"]
            print(answer) # User-facing output
        except Exception as e:
            logger.exception(f"An error occurred in embeddings: {e}")

    def get_lm_studio_models_openai(self) -> None:
        """
        Fetch and print the list of models using the OpenAI Python client.
        """
        logger.debug("Fetching models using OpenAI client.")
        try:
            models_response = self.client.models.list()
            # pprint(models_response.data) # User-facing output
            print("Available OpenAI-compatible models:") # User-facing output
            for model_data in models_response.data: # User-facing output
                print(f"  - {model_data.id}") # User-facing output
        except Exception as e:
            logger.exception(f"An error occurred in get_lm_studio_models_openai: {e}")

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
        logger.debug(f"Calling OpenAI chat completion with model: {model}")
        try:
            chat_completion = self.client.chat.completions.create(
                model=model,
                messages=self.session_history,
                temperature=0.7,
            )
            ai_message = chat_completion.choices[0].message.content
            print(ai_message) # User-facing output
            # Append the AI's response to the conversation history
            self.session_history.append({"role": "assistant", "content": ai_message})
        except Exception as e:
            logger.exception(f"An error occurred in get_chat_completion_openai: {e}")

if __name__=="__main__":
    # To see debug logs from this module for testing:
    # import logging
    # logger.setLevel(logging.DEBUG) # Set level for this module's logger
    # if logger.hasHandlers(): # Set level for its handler too
    #    logger.handlers[0].setLevel(logging.DEBUG)

    lm_studio_api_native = LMStudioAPI(openai_api=False)
    # lm_studio_api_native.get_lm_studio_models()
    # lm_studio_api_native.call_chat_completions("Tell me a joke about a computer.")
    # logger.debug(f"LM Studio Native History: {lm_studio_api_native.session_history}")

    # lm_studio_api_openai = LMStudioAPI(openai_api=True)
    # lm_studio_api_openai.get_lm_studio_models_openai()
    # lm_studio_api_openai.get_chat_completion_openai("What's the weather like in space?")
    # logger.debug(f"LM Studio OpenAI History: {lm_studio_api_openai.session_history}")

