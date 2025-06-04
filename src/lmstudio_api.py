import sys
from pprint import pformat

import requests
from openai import OpenAI

try:
    # this works when imported by llm.py
    from .library import config
    from .library.library import print_system
    from .library.logger import get_logger
except ImportError:
    # this works when running directly from /src directory
    import library.config as config
    from library.library import print_system
    from library.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__)


class LMStudioAPI:
    def __init__(
        self,
        server: str = config.DEFAULT_LMSTUDIO_SERVER,
        api_key: str = config.DEFAULT_LMSTUDIO_API_KEY,  # General API key, not directly used by client
        openai_api: bool = False,
        session_history: list = None,
    ):
        """
        Initialize the LMStudioAPI client.

        Args:
            server (str): The base URL of the LM Studio server.
            api_key (str): API key for authentication (not typically used by LM Studio for requests).
            openai_api (bool): Whether to use OpenAI-compatible endpoints.
            session_history (list): Initial conversation history (list of dicts).
        """
        self.server = server
        self.api_key = api_key
        self.openai_api = openai_api
        self.client = OpenAI(
            base_url=server + "/v1", api_key=config.DEFAULT_LMSTUDIO_API_KEY
        )
        self.session_history = (
            session_history
            if session_history is not None
            else [{"role": "system", "content": "You are a helpful assistant."}]
        )
        logger.info(
            f"LMStudioAPI initialized. OpenAI compatible: {self.openai_api}, Server: {self.server}"
        )

    def get_lm_studio_models(self) -> list | None:
        """
        Fetch and log the list of available model IDs from the LM Studio server.
        Uses OpenAI or native endpoint depending on the openai_api flag.
        Returns a list of model IDs or None if an error occurs.
        """
        if self.openai_api:
            # For OpenAI compatible, use the client's method
            return self.get_lm_studio_models_openai()

        api_endpoint = self.server + "/api/v0/models"
        logger.debug(f"Fetching models from native endpoint: {api_endpoint}")
        try:
            response = requests.get(api_endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            print_system("Available native models:")
            for model in models:
                model_id = model.get("id")
                model_type = model.get("type")
                print_system(f"  - Name: {model_id} Type: {model_type}")
            return models
        except Exception:
            logger.exception("An error occurred in get_lm_studio_models (native)")
            return None

    def get_single_model(self, model_id: str) -> dict | None:
        """
        Fetch and log details for a single model.

        Args:
            model_id (str): The model ID to fetch details for.
        Returns:
            A dictionary with model details or None if an error occurs.
        """
        # OpenAI compatible API usually doesn't have a direct equivalent for fetching non-OpenAI model details this way.
        # This method will primarily use the native LM Studio endpoint.
        if self.openai_api:
            logger.warning(
                "Fetching single model details is best via native LM Studio endpoint. OpenAI API might not provide the same level of detail for non-OpenAI models."
            )
            # Attempt to use the OpenAI client's retrieve model, though it's for OpenAI models
            try:
                model_data = self.client.models.retrieve(model_id)
                print_system(
                    f"Model details for '{model_id}' (OpenAI API):\n{pformat(model_data.to_dict())}"
                )
                return model_data.to_dict()
            except Exception:
                logger.exception(f"Error fetching model '{model_id}' via OpenAI API")
                return None

        api_endpoint = self.server + f"/api/v0/models/{model_id}"
        logger.debug(
            f"Fetching single model '{model_id}' from native endpoint: {api_endpoint}"
        )
        try:
            response = requests.get(api_endpoint, timeout=10)
            response.raise_for_status()
            model_details = response.json()
            print_system(f"Details for model '{model_id}':\n{pformat(model_details)}")
            return model_details
        except Exception:
            logger.exception("An error occurred in get_single_model (native)")
            return None

    def call_chat_completions(
        self, prompt: str, model: str = config.DEFAULT_LMSTUDIO_CHAT_MODEL
    ) -> str | None:
        """
        Send a chat completion request to the LM Studio server (native or OpenAI compatible) and log the response.
        Appends the user's prompt and AI's response to self.session_history.

        Args:
            prompt (str): The user's prompt.
            model (str): Model ID to use for chat completion.
        Returns:
            The AI's message string or None if an error occurs.
        """
        self.session_history.append({"role": "user", "content": prompt})

        if self.openai_api:
            return self.get_chat_completion_openai(
                prompt=None, model=model
            )  # prompt=None because it's already in history

        # Native LM Studio chat completions
        api_endpoint = self.server + "/api/v0/chat/completions"
        logger.debug(
            f"Calling native chat completions at: {api_endpoint} with model: {model}"
        )
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": self.session_history,
            "temperature": 0.7,
            "max_tokens": -1,  # Or a specific value like 2048
            "stream": False,
        }
        try:
            response = requests.post(
                api_endpoint, headers=headers, json=payload, timeout=60
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            logger.info(f"LMStudio Native Chat Response: {answer}")
            self.session_history.append({"role": "assistant", "content": answer})
            return answer
        except Exception:
            logger.exception("An error occurred in call_chat_completions (native)")
            # Remove the user prompt if the call failed to avoid inconsistent history
            if self.session_history and self.session_history[-1]["role"] == "user":
                self.session_history.pop()
            return None

    def completions(
        self, prompt: str, model: str = config.DEFAULT_LMSTUDIO_COMPLETION_MODEL
    ) -> str | None:
        """
        Send a text completion request to the LM Studio server (native or OpenAI compatible) and log the response.
        Appends the user's prompt and AI's response to self.session_history.

        Args:
            prompt (str): The prompt string to complete.
            model (str): Model ID to use for completion.
        Returns:
            The AI's completed text string or None if an error occurs.
        """
        # For completions, the history management might be different.
        # Typically, a completion endpoint takes a single prompt.
        # We'll add the user's prompt to history for consistency, but the payload might only use the current prompt.

        if self.openai_api:
            # OpenAI client's completion (legacy, prefer chat completions if possible)
            logger.debug(f"Calling OpenAI compatible completions with model: {model}")
            try:
                # Note: OpenAI's v1 client focuses on chat. For older completion style:
                # completion = self.client.completions.create(model=model, prompt=prompt, max_tokens=100, temperature=0.7)
                # answer = completion.choices[0].text
                # For consistency, let's adapt it to a chat-like interaction if the model supports it,
                # or acknowledge this might be a legacy call.
                # If using a true completion model, the history update might be just the prompt and response.
                # For now, let's assume we are trying to use it like a chat if possible.
                self.session_history.append({"role": "user", "content": prompt})
                chat_completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],  # Or self.session_history if model handles it well
                    temperature=0.7,
                    max_tokens=150,
                )
                answer = chat_completion.choices[0].message.content
                logger.info(f"LMStudio OpenAI Completion-Style Response: {answer}")
                self.session_history.append({"role": "assistant", "content": answer})
                return answer
            except Exception:
                logger.exception("An error occurred in completions (OpenAI compatible)")
                if self.session_history and self.session_history[-1]["role"] == "user":
                    self.session_history.pop()
                return None

        # Native LM Studio completions
        self.session_history.append(
            {"role": "user", "content": prompt}
        )  # Add to history before call
        api_endpoint = self.server + "/api/v0/completions"
        logger.debug(
            f"Calling native completions at: {api_endpoint} with model: {model}"
        )
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False,
            "stop": "\n",  # Optional stop sequence
        }
        try:
            response = requests.post(
                api_endpoint, headers=headers, json=payload, timeout=60
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["text"]
            # complete_sentence = prompt + answer # Or just answer
            logger.info(f"LMStudio Native Completion Response: {answer}")
            self.session_history.append({"role": "assistant", "content": answer})
            return answer
        except Exception:
            logger.exception("An error occurred in completions (native)")
            if self.session_history and self.session_history[-1]["role"] == "user":
                self.session_history.pop()
            return None

    def embeddings(
        self,
        input_text: str = "Some text to embed",
        model: str = config.DEFAULT_LMSTUDIO_EMBEDDING_MODEL,
    ) -> list | None:
        """
        Request embeddings for the given input text and log the embedding vector.

        Args:
            input_text (str): The text to embed.
            model (str): Model ID to use for embeddings.
        Returns:
            A list representing the embedding vector or None if an error occurs.
        """
        if self.openai_api:
            logger.debug(f"Requesting OpenAI compatible embeddings for model: {model}")
            try:
                embedding_response = self.client.embeddings.create(
                    model=model, input=input_text
                )
                embedding_vector = embedding_response.data[0].embedding
                logger.info(
                    f"LMStudio OpenAI Embedding Vector (first 5): {embedding_vector[:5]}..."
                )
                return embedding_vector
            except Exception:
                logger.exception("An error occurred in embeddings (OpenAI compatible)")
                return None

        # Native LM Studio embeddings
        api_endpoint = self.server + "/api/v0/embeddings"
        logger.debug(
            f"Requesting native embeddings from: {api_endpoint} for model: {model}"
        )
        headers = {"Content-Type": "application/json"}
        payload = {"model": model, "input": input_text}
        try:
            response = requests.post(
                api_endpoint, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()
            data = response.json()
            embedding_vector = data["data"][0]["embedding"]
            logger.info(
                f"LMStudio Native Embedding Vector (first 5): {embedding_vector[:5]}..."
            )
            return embedding_vector
        except Exception:
            logger.exception("An error occurred in embeddings (native)")
            return None

    def get_lm_studio_models_openai(self) -> list | None:
        """
        Fetch and log the list of models using the OpenAI Python client.
        Returns a list of model IDs or None if an error occurs.
        """
        logger.debug("Fetching models using OpenAI client.")
        try:
            models_response = self.client.models.list()
            model_ids = [model_data.id for model_data in models_response.data]
            print_system("Available OpenAI-compatible models (via client):")
            for model_id in model_ids:
                print_system(f"  - {model_id}")
            return model_ids
        except Exception:
            logger.exception("An error occurred in get_lm_studio_models_openai")
            return None

    def get_chat_completion_openai(
        self, prompt: str | None, model: str = config.DEFAULT_LMSTUDIO_CHAT_MODEL
    ) -> str | None:
        """
        Run a multi-turn chat completion using the OpenAI-compatible API and log the response.
        Appends the user's prompt (if provided) and AI's response to self.session_history.

        Args:
            prompt (str | None): The user's prompt. If None, assumes prompt is already in history (e.g., called by call_chat_completions).
            model (str): The model ID to use for chat.
        Returns:
            The AI's message string or None if an error occurs.
        """
        if prompt:  # Only add to history if a new prompt is given
            self.session_history.append({"role": "user", "content": prompt})

        print_system(f"Requesting OpenAI chat completion with model: {model}")
        logger.debug(
            f"Calling OpenAI chat completion with model: {model} using current session history."
        )
        try:
            chat_completion = self.client.chat.completions.create(
                model=model,
                messages=self.session_history,  # Uses the full history
                temperature=0.7,
            )
            ai_message = chat_completion.choices[0].message.content
            logger.info(f"LMStudio OpenAI Chat Response: {ai_message}")
            self.session_history.append({"role": "assistant", "content": ai_message})
            return ai_message
        except Exception:
            logger.exception("An error occurred in get_chat_completion_openai")
            # If a prompt was added for this call and it failed, remove it
            if (
                prompt
                and self.session_history
                and self.session_history[-1]["role"] == "user"
            ):
                self.session_history.pop()
            return None


def select_model(models: list) -> str | None:
    """Helper function to select a model."""
    print_system("\nAvailable models:")
    for i, model in enumerate(models):
        if isinstance(model, dict):
            # Native API returns dict
            model_name = model.get("id", "Unknown")
            model_type = model.get("type", "")
            print_system(f"{i + 1}. {model_name} (Type: {model_type})")
        else:
            # OpenAI API returns string
            print_system(f"{i + 1}. {model}")
    print_system("0. Use default model")

    try:
        choice = int(input("Select model number: "))
        if choice == 0:
            return config.DEFAULT_LMSTUDIO_CHAT_MODEL
        elif 1 <= choice <= len(models):
            if isinstance(models[choice - 1], dict):
                return models[choice - 1].get("id")
            else:
                return models[choice - 1]
        else:
            print_system("Invalid selection. Using current/default model.")
            return None
    except ValueError:
        print_system("Invalid input. Using current/default model.")
        return None


def main():
    # Simple chat interface for testing LMStudioAPI directly
    print_system("=== LM Studio API Direct Chat ===")

    # Ask about API mode
    print_system("Use OpenAI-compatible API? [yes/no] (default: no)")
    openai_choice = input().strip().lower()
    use_openai_api = openai_choice == "yes"

    api = LMStudioAPI(openai_api=use_openai_api)
    print_system(
        f"Using {'OpenAI-compatible' if use_openai_api else 'native'} API mode"
    )

    # Show available models
    print_system("\nFetching available models...")
    models = api.get_lm_studio_models()

    if not models:
        print_system("No models available. Make sure LM Studio is running.")
        sys.exit(1)

    # Initial model selection
    print_system("\nSelect initial model:")
    selected_model = select_model(models)
    if not selected_model:
        selected_model = config.DEFAULT_LMSTUDIO_CHAT_MODEL
    print_system(f"Using model: {selected_model}")

    # Main chat loop
    print_system("\nStarting chat with LM Studio. Type 'exit' to quit.")
    print_system(
        "Commands: 'chat', 'completion', 'select', 'models', 'history', 'clear', 'mode', or 'exit'\n"
    )

    while True:
        print_system(f"\nCurrent model: {selected_model}")
        print_system(f"API mode: {'OpenAI-compatible' if api.openai_api else 'Native'}")
        cmd = input("Command: ").strip().lower()

        if cmd == "exit":
            print_system("Goodbye!")
            break

        elif cmd == "chat":
            user_prompt = input("Your message: ")
            print_system("\nThinking...")
            response = api.call_chat_completions(
                prompt=user_prompt, model=selected_model
            )
            if response:
                print_system(f"\nAssistant: {response}")
            else:
                print_system("Error: Failed to get response")

        elif cmd == "completion":
            user_prompt = input("Your prompt: ")
            print_system("\nGenerating...")
            response = api.completions(prompt=user_prompt, model=selected_model)
            if response:
                print_system(f"\nCompletion: {response}")
            else:
                print_system("Error: Failed to get completion")

        elif cmd == "select":
            # Allow user to change model during session
            new_model = select_model(models)
            if new_model:
                selected_model = new_model
                print_system(f"Switched to model: {selected_model}")
            else:
                print_system("Model selection cancelled.")

        elif cmd == "models":
            # Re-fetch and display available models
            print_system("\nRefreshing model list...")
            models = api.get_lm_studio_models()
            if not models:
                print_system("Failed to refresh models.")

        elif cmd == "history":
            print_system("\n=== Conversation History ===")
            for msg in api.session_history:
                role = msg["role"].capitalize()
                content = (
                    msg["content"][:100] + "..."
                    if len(msg["content"]) > 100
                    else msg["content"]
                )
                print_system(f"{role}: {content}")
            print_system("=========================\n")

        elif cmd == "clear":
            api.session_history = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            print_system("Conversation history cleared.")

        elif cmd == "mode":
            # Switch between OpenAI and native API modes
            print_system("Switch to which API mode? [openai/native]")
            mode_choice = input().strip().lower()
            if mode_choice == "openai":
                api = LMStudioAPI(openai_api=True)
                print_system("Switched to OpenAI-compatible API mode")
            elif mode_choice == "native":
                api = LMStudioAPI(openai_api=False)
                print_system("Switched to native API mode")
            else:
                print_system("Invalid choice. Mode unchanged.")
                continue

            # Re-fetch models after switching mode
            print_system("\nFetching models for new mode...")
            models = api.get_lm_studio_models()
            if not models:
                print_system("No models available in this mode.")

        else:
            print_system(
                "Unknown command. Use: 'chat', 'completion', 'select', 'models', 'history', 'clear', 'mode', or 'exit'"
            )


if __name__ == "__main__":
    main()
