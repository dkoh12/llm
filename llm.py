from ollama_api import OllamaAPI
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
            self.api = OllamaAPI(**kwargs)
            self.provider = "ollama"
        elif provider == "lmstudio":
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
        print(" ", end='\r', flush=True)

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
                # Use ollama_chat if available
                if model:
                    self.api.ollama_chat(model=model)
                else:
                    self.api.ollama_chat()
            elif self.provider == "lmstudio":
                # LMStudio expects a list of messages
                messages = [
                    {"role": "user", "content": prompt}
                ]
                if model:
                    self.api.call_chat_completions(messages, model=model)
                else:
                    self.api.call_chat_completions(messages)
            else:
                print("Unknown provider.")
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
                    self.api.text_completion(model=model)
                else:
                    self.api.text_completion()
            elif self.provider == "lmstudio":
                if model:
                    self.api.completions(prompt, model=model)
                else:
                    self.api.completions(prompt)
            else:
                print("Unknown provider.")
        finally:
            stop_event.set()
            progress_thread.join()

if __name__ == "__main__":
    # Example usage
    print("Choose provider: [ollama/lmstudio]")
    provider = input().strip().lower()
    agent = LLMUnifiedAgent(provider=provider)

    print("Type 'chat' for chat, 'completion' for text completion, or 'exit' to quit.")
    while True:
        cmd = input("Command: ").strip().lower()
        if cmd == "exit":
            break
        elif cmd == "chat":
            prompt = input("Your message: ")
            agent.chat(prompt)
        elif cmd == "completion":
            prompt = input("Your prompt: ")
            agent.text_completion(prompt)
        else:
            print("Unknown command.")