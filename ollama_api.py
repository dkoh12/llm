import pprint
from openai import OpenAI
import ollama
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

# https://github.com/ollama/ollama/blob/main/docs/api.md
# https://github.com/ollama/ollama-python

# main apis are generate, chat, and models


def ollama_chat():
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the purpose of life?"},
        ],
    )

    print(response["message"]["content"])


# using raw ollama api to test vision model
def multimodal_1():
    res = ollama.chat(
        model="llava:7b",
        messages=[
            {
                "role": "user",
                "content": "Describe the image",
                "images": ['./tesla-model-y-top.jpg']
            }
        ],
    )

    print(res["message"]["content"])

def multimodal_2():
    # not necessary to use file api
    with open('tesla-model-y-top.jpg', 'rb') as f:
        res = ollama.chat(
            model="llava:7b",
            messages=[
                {
                    "role": "user",
                    "content": "What is the brand of the car?",
                    "images": [f.read()]
                }
            ],
        )

    print(res["message"]["content"])

def text_completion():
    result = ollama.generate(
        model="codellama:latest",
        prompt="// A c function to reverse a string",
    )

    print(result["response"])

# ------------------------------------------

def openai_chat(client):
    messages=[
        {
            "role": "user", 
            "content": "Say this is a test"
        }
    ]
    try:
        chat_completion = client.chat.completions.create(
            model="llama3.2:latest",
            messages=messages,
            temperature=0.7,
        )
        print(chat_completion.choices[0].message.content)
    except Exception as e:
        print(e)

# model has to be multimodal and available in `ollama list`
def image(client):
    try:
        response = client.chat.completions.create(
            model="llava:7b",
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


# model has to be available in `ollama list`
def get_chat_completion_openai(client):
    try:
        chat_completion = client.chat.completions.create(        
            model="llama3.2:latest",
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

# works but takes some time.
def autogen():

    config_list = [
        {
            "model": "codellama:latest",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        }
    ]

    assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})

    user_proxy = UserProxyAgent("user_proxy", code_execution_config=
                                {
                                    "work_dir": "coding",
                                    "use_docker": False
                                })
    
    user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")

def conversable_agent():
    config_list = [
        {
            "model": "codellama:latest",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        }
    ]

    my_agent = ConversableAgent("helpful_agent", 
                                llm_config={"config_list": config_list},
                                system_message="You are a poetic AI assistant, respond in rhymes.")

    my_agent.run("In one sentence, what's the big deal about AI?")
 
if __name__=="__main__":
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama" # required but ignored
    )

    # ollama_chat()

    # multimodal_1()

    # multimodal_2()

    # text_completion()

    # openai_chat(client)

    # image(client)

    # get_chat_completion_openai(client)

    # autogen()

    conversable_agent()