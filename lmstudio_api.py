import requests
import json
from pprint import pprint
from openai import OpenAI

class LMStudioAPI:
    def __init__(self, server: str = "http://localhost:1234", api_key: str = "", openai_api: bool = False):
        self.server = server
        self.api_key = api_key
        self.openai_api = openai_api
        self.client = OpenAI(
            base_url=server + "/v1",
            api_key="lm-studio"  # required but ignored
        )

    def get_lm_studio_models(self) -> None:
        if self.openai_api:
            api_endpoint = self.server + "/v1/models"
        else:
            api_endpoint = self.server + "/api/v0/models"

        try:
            response = requests.get(api_endpoint)
            if response.status_code == 200:
                data = response.json()
                for model in data["data"]:
                    print(model["id"])
        except Exception as e:
            print(f"Error: {e}")

    def get_single_model(self, model: str) -> None:
        if self.openai_api:
            api_endpoint = self.server + f"/v1/models/{model}"
        else:
            api_endpoint = self.server + f"/api/v0/models/{model}"

        try:
            response = requests.get(api_endpoint)
            if response.status_code == 200:
                data = response.json()
                pprint(data)
        except Exception as e:
            print(f"Error: {e}")

    def call_chat_completions(self, message: list, model: str = "llama-3.2-3b-instruct") -> None:
        if self.openai_api:
            api_endpoint = self.server + "/v1/chat/completions"
        else:
            api_endpoint = self.server + "/api/v0/chat/completions"

        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": model,
            "messages": message,
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
        except Exception as e:
            print(f"Error: {e}")

    def completions(self, prompt: str, model: str = "llama-3.2-3b-instruct") -> None:
        if self.openai_api:
            api_endpoint = self.server + "/v1/completions"
        else:
            api_endpoint = self.server + "/api/v0/completions"

        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": model,
            "prompt": prompt,
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
                complete_sentence = payload["prompt"] + answer
                print(complete_sentence)
        except Exception as e:
            print(f"Error: {e}")

    def embeddings(self, input_text: str = "Some text to embed", model: str = "text-embedding-nomic-embed-text-v1.5") -> None:
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
                error_message = data["error"]["message"]
                print(error_message)
            else:
                print("Error: ", response.status_code)
                print(response.text)
        except Exception as e:
            print(f"Error: {e}")

    def get_lm_studio_models_openai(self) -> None:
        try:
            models= self.client.models.list()
            pprint(models.data)
        except Exception as e:
            print(f"Error: {e}")

    def get_chat_completion_openai(self) -> None:

        """
        messages=[
                {
                    "role": "system",
                    "content": "Always answer in rhymes."
                },
                {
                    "role": "user",
                    "content": "Introduce yourself.",
                }
            ],
        """


        chat_completion = self.client.chat.completions.create(        
            model="llama-3.2-3b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Who won the world series in 2020?",
                },
                {
                    "role": "assistant",
                    "content": "The Los Angeles Dodgers won the 2020 World Series."
                },
                {
                    "role": "user",
                    "content": "Where was it played?",
                }
            ],
            temperature=0.7,
        )

        print(chat_completion.choices[0].message.content)



if __name__=="__main__":
    server = "http://localhost:1234"
    api_key = "" # nothing

    lm_studio_api = LMStudioAPI(server=server, api_key=api_key, openai_api=False)

    # lm_studio_api.get_lm_studio_models()


    client = OpenAI(
        base_url=server + "/v1",
        api_key="lm-studio" # required but ignored
    )

    # lm_studio_api.get_lm_studio_models_openai()

    lm_studio_api.get_chat_completion_openai()

    #get_single_model(server, model="llama-3.2-3b-instruct")

    history = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]

    """
    new_message = {
        "role": "user",
        "content": "What is the capital of the United States?"
    }"
    """

    # seems like this is better than completion
    new_message = {
        "role": "user",
        "content": "the meaning of life is"
    }

    history.append(new_message)
    #call_chat_completions(server, history)  

    prompt = "The meaning of life is"
    #completions(server, prompt)

    #embeddings(server)

