import pprint
from openai import OpenAI
import ollama
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

class OllamaAPI:
    def __init__(self, openai_base_url="http://localhost:11434/v1", api_key="ollama"):
        self.client = OpenAI(
            base_url=openai_base_url,
            api_key=api_key
        )

    def ollama_chat(self, model="llama3.2"):
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the purpose of life?"},
            ],
        )
        print(response["message"]["content"])

    def multimodal_1(self, model="llava:7b"):
        res = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Describe the image",
                    "images": ['./tesla-model-y-top.jpg']
                }
            ],
        )
        print(res["message"]["content"])

    def multimodal_2(self, model="llava:7b"):
        with open('tesla-model-y-top.jpg', 'rb') as f:
            res = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the brand of the car?",
                        "images": [f.read()]
                    }
                ],
            )
        print(res["message"]["content"])

    def text_completion(self, model="codellama:latest"):
        result = ollama.generate(
            model=model,
            prompt="// A c function to reverse a string",
        )
        print(result["response"])

    def openai_chat(self, model="llama3.2:latest"):
        messages = [
            {
                "role": "user",
                "content": "Say this is a test"
            }
        ]
        try:
            chat_completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
            )
            print(chat_completion.choices[0].message.content)
        except Exception as e:
            print(e)

    def image(self, model="llava:7b"):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": "./tesla-model-y-top.jpg"},
                        ]
                    }
                ],
                max_tokens=300,
            )
            print(response)
        except Exception as e:
            print(e)

    def get_chat_completion_openai(self, model="llama3.2:latest"):
        try:
            chat_completion = self.client.chat.completions.create(
                model=model,
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
        except Exception as e:
            print(e)


def autogen(model="codellama:latest"):
    config_list = [
        {
            "model": model,
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        }
    ]

    assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})

    user_proxy = UserProxyAgent("user_proxy", code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    })
    
    user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")

def conversable_agent(model="codellama:latest"):
    config_list = [
        {
            "model": model,
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        }
    ]

    my_agent = ConversableAgent(
        "helpful_agent", 
        llm_config={"config_list": config_list},
        system_message="You are a poetic AI assistant, respond in rhymes."
    )

    my_agent.run("In one sentence, what's the big deal about AI?")
 
if __name__=="__main__":
    api = OllamaAPI()

    print(1)

    # api.ollama_chat()
    # api.multimodal_1()
    # api.multimodal_2()
    # api.text_completion()
    # api.openai_chat()
    api.image()
    api.get_chat_completion_openai()
    # autogen()
    # conversable_agent()