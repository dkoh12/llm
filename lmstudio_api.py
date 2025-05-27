import requests
import json
from pprint import pprint
from openai import OpenAI

# https://lmstudio.ai/docs/app/api/tools

OPENAI_API = False #True

"""
Lists currently loaded models
"""
# GET /v1/models
def get_lm_studio_models(server):
    if OPENAI_API:
        api_endpoint = server + "/v1/models"
    else:
        api_endpoint = server + "/api/v0/models"

    try:
        response = requests.get(api_endpoint)

        if response.status_code == 200:
            data = response.json()
            for model in data["data"]:
                print(model["id"])
    except Exception as e:
        print(f"Error: {e}")


"""
Lists a specific model
"""
# GET /v1/models/{model}
def get_single_model(server, model):
    if OPENAI_API:
        api_endpoint = server + f"/v1/models/{model}"
    else:
        api_endpoint = server + f"/api/v0/models/{model}"

    try:
        response = requests.get(api_endpoint)

        if response.status_code == 200:
            data = response.json()
            pprint(data)
    except Exception as e:
        print(f"Error: {e}")


"""
Send a chat history and receive the assistant's response
Generates completions for chat messages
"""
# POST /v1/chat/completions
def call_chat_completions(server, message):
    if OPENAI_API:
        api_endpoint = server + "/v1/chat/completions"
    else:
        api_endpoint = server + "/api/v0/chat/completions"
        
    # 'Authorization': f'Bearer {api_key}',
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "llama-3.2-3b-instruct",
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
        return f"Error: {e}"

"""
Send a prompt and receive completions"""
# POST /v1/completions
def completions(server, prompt):
    if OPENAI_API:
        api_endpoint = server + "/v1/completions"
    else:
        api_endpoint = server + "/api/v0/completions"
    
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "llama-3.2-3b-instruct",
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
        return f"Error: {e}"

"""
Send a string or array of strings and 
get an array of text embeddings (integer token IDs)

This api only works with models that have the text-embedding capability
"""
# POST /v1/embeddings
def embeddings(server):
    if OPENAI_API:
        api_endpoint = server + "/v1/embeddings"
    else:
        api_endpoint = server + "/api/v0/embeddings"
    
    # 'Authorization': f'Bearer {api_key}',
    headers = {
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "text-embedding-nomic-embed-text-v1.5",
        "input": "Some text to embed"
    }
    
    try:
        response = requests.post(api_endpoint, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            answer = data["data"][0]["embedding"]
            print(answer)
        elif response.status_code == 404:
            # hits this when the model does not have the text-embedding capability
            data = response.json()
            error_message = data["error"]["message"]
            print(error_message)
        else:
            # response status code = 4.04
            print("Error: ", response.status_code)
            print(response.text)
    except Exception as e:
        return f"Error: {e}"


def get_lm_studio_models_openai(client):
    try:
        models= client.models.list()
        pprint(models.data)
    except Exception as e:
        print(f"Error: {e}")


def get_chat_completion_openai(client):

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


    chat_completion = client.chat.completions.create(        
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

    # get_lm_studio_models(server)


    client = OpenAI(
        base_url=server + "/v1",
        api_key="lm-studio" # required but ignored
    )

    # get_lm_studio_models_openai(client)

    get_chat_completion_openai(client)

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


# todo - build a chat bot
# todo - build a code copilot


