import pprint
from openai import OpenAI
import ollama
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

class OllamaAPI:
    def __init__(self, openai_base_url: str = "http://localhost:11434/v1", api_key: str = "ollama", session_history: list = None):
        """
        Initialize the OllamaAPI client.

        Args:
            openai_base_url (str): The base URL for the Ollama OpenAI-compatible API.
            api_key (str): API key for authentication (default is "ollama").
            session_history (list): Initial conversation history (list of dicts).
        """
        self.client = OpenAI(
            base_url=openai_base_url,
            api_key=api_key
        )
        self.session_history = session_history if session_history is not None else [{"role": "system", "content": "You are a helpful assistant."}]

    def ollama_chat(self, prompt: str, model: str = "llama3.2"):
        """
        Run a chat completion using the Ollama API and print the response.
        Appends the user's prompt and AI's response to the session history.

        Args:
            prompt (str): The user's prompt.
            model (str): The model ID to use for chat.
        """
        # Append the user's prompt to the session history
        self.session_history.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(
                model=model,
                messages=self.session_history,
            )
            ai_message = response["message"]["content"]
            print(ai_message)

            # Append the AI's response to the session history
            self.session_history.append({"role": "assistant", "content": ai_message})
        except Exception as e:
            print(f"Error in ollama_chat: {e}")

    def multimodal_1(self, model: str = "llava:7b"):
        """
        Run a multimodal chat completion with an image file path and print the response.

        Args:
            model (str): The model ID to use for multimodal chat.
        """
        res = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Describe the image",
                    "images": ['./images/tesla-model-y-top.jpg']
                }
            ],
        )
        print(res["message"]["content"])

    def multimodal_2(self, model: str = "llava:7b"):
        """
        Run a multimodal chat completion by reading an image as bytes and print the response.

        Args:
            model (str): The model ID to use for multimodal chat.
        """
        with open('images/tesla-model-y-top.jpg', 'rb') as f:
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

    def text_completion(self, prompt: str, model: str = "codellama:latest"):
        """
        Generate a text completion using the Ollama API and print the response.

        Args:
            prompt (str): The prompt string to complete.
            model (str): The model ID to use for text completion.
        """
        result = ollama.generate(
            model=model,
            prompt=prompt,
        )
        print(result["response"])

    def openai_chat(self, prompt: str, model: str = "llama3.2:latest"):
        """
        Run a chat completion using the OpenAI-compatible API and print the response.

        Args:
            prompt (str): The user's prompt.
            model (str): The model ID to use for chat.
        """
        messages = [
            {
                "role": "user",
                "content": prompt
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

    def image(self, model: str = "llava:7b"):
        """
        Run a multimodal chat completion using the OpenAI-compatible API with an image and print the response.

        Args:
            model (str): The model ID to use for multimodal chat.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": "./images/tesla-model-y-top.jpg"},
                        ]
                    }
                ],
                max_tokens=300,
            )
            print(response)
        except Exception as e:
            print(e)

    def get_chat_completion_openai(self, prompt: str, model: str = "llama3.2:latest"):
        """
        Run a multi-turn chat completion using the OpenAI-compatible API and print the response.
        Appends the user's prompt and AI's response to self.history.

        Args:
            prompt (str): The user's prompt.
            model (str): The model ID to use for chat.
        """
        # Append the user's prompt to the conversation history
        self.session_history.append({"role": "user", "content": prompt})
        try:
            chat_completion = self.client.chat.completions.create(
                model=model,
                messages=self.session_history,
                temperature=0.7,
            )
            ai_message = chat_completion.choices[0].message.content
            print(ai_message)
            # Append the AI's response to the conversation history
            self.session_history.append({"role": "assistant", "content": ai_message})
        except Exception as e:
            print(e)


def autogen(model: str = "codellama:latest"):
    """
    Run an autogen workflow using the AssistantAgent and UserProxyAgent.

    Args:
        model (str): The model ID to use for the agents.
    """
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

def conversable_agent(model: str = "codellama:latest"):
    """
    Run a ConversableAgent with a poetic system message and print the response.

    Args:
        model (str): The model ID to use for the agent.
    """
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
    
    # api.ollama_chat(prompt="Hello, who won the world series in 2020?")
    # print(f"Ollama Session History: {api.session_history}")
    # api.multimodal_1()
    # api.multimodal_2()
    # api.text_completion(prompt="// A python function to add two numbers")
    # api.openai_chat(prompt="What is the capital of Canada?")
    # api.get_chat_completion_openai(prompt="And what is its population?")
    # print(f"Ollama Session History after OpenAI call: {api.session_history}")
    # api.image()
    # autogen()
    # conversable_agent()