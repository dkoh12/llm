from ollama_api import OllamaAPI
from library import print_system
from lmstudio_api import LMStudioAPI
import threading
import time
import sys

class LLMUnifiedAgent:
    """
    A unified agent interface to interact with either Ollama or LM Studio APIs.
    """
    def __init__(self, provider: str = "ollama", **kwargs):
        """
        Initialize the unified agent.

        Args:
            provider (str): "ollama" or "lmstudio"
            **kwargs: Arguments to pass to the underlying API class.
        """
        if provider == "ollama":
            # Pass kwargs to OllamaAPI constructor if it accepts them
            # For now, assuming default OllamaAPI initialization or specific args if needed
            self.api = OllamaAPI(**kwargs)
            self.provider = "ollama"
        elif provider == "lmstudio":
            # Pass kwargs to LMStudioAPI constructor
            self.api = LMStudioAPI(**kwargs)
            self.provider = "lmstudio"
        else:
            raise ValueError("Provider must be 'ollama' or 'lmstudio'.")

    def _progress_bar(self, stop_event: threading.Event):
        spinner = ['|', '/', '-', '\\']
        idx = 0
        print("Waiting for response ", end='', flush=True)
        while not stop_event.is_set():
            print(spinner[idx % len(spinner)], end='\b', flush=True)
            idx += 1
            time.sleep(0.1)
        print(" ", end='\r', flush=True) # Clear the spinner

    def get_models(self) -> list | None:
            """
            Get available models from the current provider.
            
            Returns:
                list | None: A list of models or None if error occurred.
            """
            try:
                if self.provider == "ollama":
                    return self.api.get_ollama_models()
                elif self.provider == "lmstudio":
                    return self.api.get_lm_studio_models()
                else:
                    return None
            except Exception as e:
                print_system(f"Error during getting models: {e}")

    def chat(self, prompt: str, model: str = None):
        """
        Send a chat prompt to the selected provider and print the response.

        Args:
            prompt (str): The user message.
            model (str): The model to use (optional).
        """
        stop_event = threading.Event()
        progress_thread = threading.Thread(target=self._progress_bar, args=(stop_event,))
        progress_thread.start()
        try:
            if self.provider == "ollama":
                if model:
                    self.api.ollama_chat(prompt=prompt, model=model)
                else:
                    self.api.ollama_chat(prompt=prompt)
            elif self.provider == "lmstudio":
                if model:
                    self.api.call_chat_completions(prompt=prompt, model=model)
                else:
                    self.api.call_chat_completions(prompt=prompt)
            else:
                print_system("Unknown provider.")
        except Exception as e:
            print_system(f"Error during chat: {e}")
        finally:
            stop_event.set()
            progress_thread.join()

    def text_completion(self, prompt: str, model: str = None):
        """
        Send a text completion prompt to the selected provider and print the response.

        Args:
            prompt (str): The prompt string.
            model (str): The model to use (optional).
        """
        stop_event = threading.Event()
        progress_thread = threading.Thread(target=self._progress_bar, args=(stop_event,))
        progress_thread.start()
        try:
            if self.provider == "ollama":
                if model:
                    self.api.text_completion(prompt=prompt, model=model)
                else:
                    self.api.text_completion(prompt=prompt)
            elif self.provider == "lmstudio":
                if model:
                    self.api.completions(prompt=prompt, model=model)
                else:
                    self.api.completions(prompt=prompt)
            else:
                print_system("Unknown provider.")
        except Exception as e:
            print_system(f"Error during text completion: {e}")
        finally:
            stop_event.set()
            progress_thread.join()

if __name__ == "__main__":
    print_system("Choose provider: [ollama/lmstudio]")
    provider_choice = input().strip().lower()
    
    agent_kwargs = {}
    if provider_choice == "lmstudio":
        print_system("Use OpenAI-compatible API for LM Studio? [yes/no] (default: no)")
        openai_choice = input().strip().lower()
        agent_kwargs['openai_api'] = True if openai_choice == 'yes' else False

    try:
        agent = LLMUnifiedAgent(provider=provider_choice, **agent_kwargs)
    except ValueError as e:
        print_system(str(e))
        sys.exit(1)


    while True:
        print_system("Type 'chat' for chat, 'completion' for text completion, 'models' to view existing models, or 'exit' to quit.")
        cmd = input("Command: ").strip().lower()
        if cmd == "exit":
            break
        elif cmd == "chat":
            user_prompt = input("Your message: ")
            agent.chat(prompt=user_prompt)
        elif cmd == "completion":
            user_prompt = input("Your prompt: ")
            agent.text_completion(prompt=user_prompt)
        elif cmd == "models":
            agent.get_models()
        else:
            print_system("Unknown command.")