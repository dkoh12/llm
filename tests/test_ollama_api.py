import unittest
from ollama_api import OllamaAPI


class TestOllamaAPI(unittest.TestCase):
    def setUp(self):
        # You can customize session_history for each test if needed
        self.history = [{"role": "system", "content": "You are a helpful assistant."}]
        self.api = OllamaAPI(session_history=self.history)

    def test_ollama_chat(self):
        # This will actually call the Ollama server if running
        self.api.ollama_chat("Say hello!")
        self.assertTrue(
            any("assistant" == msg["role"] for msg in self.api.session_history)
        )

    def test_text_completion(self):
        self.api.text_completion(prompt="What is the capital of France?")
        # No assertion, just checks that it runs without error

    def test_openai_chat(self):
        self.api.openai_chat(prompt="Say this is a test")
        # No assertion, just checks that it runs without error

    def test_get_chat_completion_openai(self):
        self.api.get_chat_completion_openai(prompt="Who won the world series in 2020?")
        # No assertion, just checks that it runs without error


if __name__ == "__main__":
    unittest.main()
