import unittest
from unittest.mock import patch, MagicMock
from lmstudio_api import LMStudioAPI # Assuming your LMStudioAPI class is in lmstudio_api.py

# Helper to create a mock requests.Response object
def _mock_response(status=200, content=None, json_data=None, raise_for_status=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.content = content
    if json_data:
        mock_resp.json = MagicMock(return_value=json_data)
    if raise_for_status:
        mock_resp.raise_for_status = MagicMock(side_effect=raise_for_status)
    return mock_resp

# A mock response object for openai.ChatCompletion (similar to Ollama's test)
class MockChatCompletionChoice:
    def __init__(self, content):
        self.message = MagicMock()
        self.message.content = content

class MockChatCompletion:
    def __init__(self, content):
        self.choices = [MockChatCompletionChoice(content)]

class TestLMStudioAPI(unittest.TestCase):
    def setUp(self):
        self.initial_history = [{"role": "system", "content": "You are a helpful assistant."}]
        # Using openai_api=False for tests that use requests.post/get
        # and openai_api=True for tests that use self.api.client
        self.api_native = LMStudioAPI(openai_api=False, session_history=list(self.initial_history))
        # self.api_native.session_history = list(self.initial_history) # Already handled by constructor

        self.api_openai_compatible = LMStudioAPI(openai_api=True, session_history=list(self.initial_history))
        # self.api_openai_compatible.session_history = list(self.initial_history) # Already handled by constructor


    @patch('lmstudio_api.requests.get') # Path to requests.get used in your lmstudio_api module
    def test_get_lm_studio_models_native(self, mock_get):
        mock_json_response = {"data": [{"id": "model1"}, {"id": "model2"}]}
        mock_get.return_value = _mock_response(json_data=mock_json_response)
        
        self.api_native.get_lm_studio_models() # This method prints, so we mainly check the call
        
        mock_get.assert_called_once_with(self.api_native.server + "/api/v0/models")

    @patch('lmstudio_api.requests.get')
    def test_get_single_model_native(self, mock_get):
        model_id = "llama-3.2-3b-instruct"
        mock_json_response = {"id": model_id, "details": "some details"}
        mock_get.return_value = _mock_response(json_data=mock_json_response)

        self.api_native.get_single_model(model_id)
        mock_get.assert_called_once_with(self.api_native.server + f"/api/v0/models/{model_id}")

    @patch('lmstudio_api.requests.post')
    def test_call_chat_completions_native(self, mock_post):
        mock_response_content = "Life is 42, mocked."
        mock_json_response = {
            "choices": [{"message": {"content": mock_response_content}}]
        }
        mock_post.return_value = _mock_response(json_data=mock_json_response)
        
        test_prompt = "Hello, what is the meaning of life?"
        # Reset session_history for this specific test if needed, or ensure setUp provides a fresh copy
        self.api_native.session_history = list(self.initial_history)
        self.api_native.call_chat_completions(prompt=test_prompt, model="test-model")
        
        expected_history_before_api_call = self.initial_history + [{"role": "user", "content": test_prompt}]
        expected_payload = {
            "model": "test-model",
            "messages": expected_history_before_api_call,
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }
        mock_post.assert_called_once_with(
            self.api_native.server + "/api/v0/chat/completions",
            headers={'Content-Type': 'application/json'},
            json=expected_payload
        )
        
        expected_history_after_call = expected_history_before_api_call + [{"role": "assistant", "content": mock_response_content}]
        self.assertEqual(self.api_native.session_history, expected_history_after_call)

    @patch('lmstudio_api.requests.post')
    def test_completions_native(self, mock_post):
        test_prompt = "The capital of France is"
        mock_response_text = " Paris, mocked."
        mock_json_response = {
            "choices": [{"text": mock_response_text}]
        }
        mock_post.return_value = _mock_response(json_data=mock_json_response)
        
        # Reset session_history for this specific test
        self.api_native.session_history = list(self.initial_history)
        self.api_native.completions(prompt=test_prompt, model="test-completion-model")

        expected_payload = {
            "model": "test-completion-model",
            "prompt": test_prompt,
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False,
            "stop": "\n"
        }
        mock_post.assert_called_once_with(
            self.api_native.server + "/api/v0/completions",
            headers={'Content-Type': 'application/json'},
            json=expected_payload
        )
        # Assert history if completions is supposed to update it.
        expected_history_after_call = self.initial_history + \
                                      [{"role": "user", "content": test_prompt},
                                       {"role": "assistant", "content": mock_response_text}]
        self.assertEqual(self.api_native.session_history, expected_history_after_call)


    @patch.object(LMStudioAPI, 'client', new_callable=MagicMock) # Mocking the client instance
    def test_get_chat_completion_openai_compatible(self, mock_openai_client_on_instance):
        # Re-initialize api_openai_compatible to use the MagicMock client for this test
        # And ensure it gets a fresh copy of initial_history
        self.api_openai_compatible = LMStudioAPI(openai_api=True, session_history=list(self.initial_history))
        self.api_openai_compatible.client = mock_openai_client_on_instance # Assign the mock
        # self.api_openai_compatible.session_history = list(self.initial_history) # Handled by constructor


        mock_response_content = "The mocked World Series winner is Team Mock."
        mock_openai_client_on_instance.chat.completions.create.return_value = MockChatCompletion(mock_response_content)
        
        test_prompt = "Who won the world series in 2020?"
        self.api_openai_compatible.get_chat_completion_openai(prompt=test_prompt, model="test-openai-model")
        
        expected_history_call = self.initial_history + [{"role": "user", "content": test_prompt}]
        mock_openai_client_on_instance.chat.completions.create.assert_called_once_with(
            model="test-openai-model",
            messages=expected_history_call,
            temperature=0.7
        )
        
        expected_history_after_call = expected_history_call + [{"role": "assistant", "content": mock_response_content}]
        self.assertEqual(self.api_openai_compatible.session_history, expected_history_after_call)

if __name__ == "__main__":
    unittest.main()