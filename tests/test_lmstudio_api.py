import unittest
from lmstudio_api import LMStudioAPI


class TestLMStudioAPI(unittest.TestCase):
    def setUp(self):
        self.api = LMStudioAPI(openai_api=False)

    def test_get_lm_studio_models(self):
        self.api.get_lm_studio_models()
        # No assertion, just checks that it runs without error

    def test_get_single_model(self):
        self.api.get_single_model("llama-3.2-3b-instruct")
        # No assertion, just checks that it runs without error

    def test_call_chat_completions(self):
        self.api.call_chat_completions(prompt="Hello, what is the meaning of life?")
        self.assertTrue(any("assistant" == msg["role"] for msg in self.api.history))

    def test_completions(self):
        self.api.completions(prompt="The capital of France is")
        self.assertTrue(any("assistant" == msg["role"] for msg in self.api.history))

    def test_get_chat_completion_openai(self):
        self.api.get_chat_completion_openai(prompt="Who won the world series in 2020?")
        self.assertTrue(any("assistant" == msg["role"] for msg in self.api.history))


if __name__ == "__main__":
    unittest.main()
