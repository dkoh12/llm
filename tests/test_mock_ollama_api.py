import unittest
from unittest.mock import MagicMock, patch

from ollama_api import OllamaAPI  # Assuming your OllamaAPI class is in ollama_api.py


# A mock response object for openai.ChatCompletion
class MockChatCompletionChoice:
    def __init__(self, content):
        self.message = MagicMock()
        self.message.content = content


class MockChatCompletion:
    def __init__(self, content):
        self.choices = [MockChatCompletionChoice(content)]


class TestOllamaAPI(unittest.TestCase):
    def setUp(self):
        self.initial_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        # Pass a copy of the history to avoid modification across tests if not intended
        self.api = OllamaAPI(session_history=list(self.initial_history))

    @patch(
        "ollama_api.ollama.chat"
    )  # Path to ollama.chat within your ollama_api module
    def test_ollama_chat(self, mock_ollama_chat):
        # Configure the mock to return a sample response
        mock_response_content = "Hello from mocked Ollama!"
        mock_ollama_chat.return_value = {"message": {"content": mock_response_content}}

        test_prompt = "Say hello!"
        self.api.ollama_chat(prompt=test_prompt)

        # Assert ollama.chat was called correctly
        expected_messages = self.initial_history + [
            {"role": "user", "content": test_prompt}
        ]
        mock_ollama_chat.assert_called_once_with(
            model="llama3.2", messages=expected_messages
        )

        # Assert session_history is updated
        expected_history_after_call = expected_messages + [
            {"role": "assistant", "content": mock_response_content}
        ]
        self.assertEqual(self.api.session_history, expected_history_after_call)

    @patch("ollama_api.ollama.generate")  # Path to ollama.generate
    def test_text_completion(self, mock_ollama_generate):
        mock_response_content = "Paris."
        mock_ollama_generate.return_value = {"response": mock_response_content}

        test_prompt = "What is the capital of France?"
        self.api.text_completion(prompt=test_prompt)

        mock_ollama_generate.assert_called_once_with(
            model="codellama:latest", prompt=test_prompt
        )
        # If text_completion modified history, assert that too. Currently, it doesn't.

    @patch.object(OllamaAPI, "client")  # Mocking the client instance on OllamaAPI
    def test_openai_chat(self, mock_openai_client):
        mock_response_content = "This is a mocked test response."
        # Configure the client's chat.completions.create method
        mock_openai_client.chat.completions.create.return_value = MockChatCompletion(
            mock_response_content
        )

        test_prompt = "Say this is a test"
        self.api.openai_chat(prompt=test_prompt)

        expected_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": test_prompt},
        ]
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model="llama3.2:latest", messages=expected_messages, temperature=0.7
        )

    @patch.object(OllamaAPI, "client")
    def test_get_chat_completion_openai(self, mock_openai_client):
        mock_response_content = "The LA Dodgers."
        mock_openai_client.chat.completions.create.return_value = MockChatCompletion(
            mock_response_content
        )

        test_prompt = "Who won the world series in 2020?"
        self.api.get_chat_completion_openai(prompt=test_prompt)

        expected_messages_call = self.initial_history + [
            {"role": "user", "content": test_prompt}
        ]
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model="llama3.2:latest", messages=expected_messages_call, temperature=0.7
        )

        expected_history_after_call = expected_messages_call + [
            {"role": "assistant", "content": mock_response_content}
        ]
        self.assertEqual(self.api.session_history, expected_history_after_call)


if __name__ == "__main__":
    unittest.main()
