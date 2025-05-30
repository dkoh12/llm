import pprint
from openai import OpenAI
import ollama
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from logger import get_logger
import config

logger = get_logger(__name__)

class OllamaAPI:
    def __init__(self, 
                 openai_base_url: str = config.DEFAULT_OLLAMA_BASE_URL, 
                 api_key: str = config.DEFAULT_OLLAMA_API_KEY, 
                 session_history: list = None):
        """
        Initialize the OllamaAPI client.

        Args:
            openai_base_url (str): The base URL for the Ollama OpenAI-compatible API.
            api_key (str): API key for authentication.
            session_history (list): Initial conversation history (list of dicts).
        """
        self.client = OpenAI(
            base_url=openai_base_url,
            api_key=api_key
        )
        self.session_history = session_history if session_history is not None else [{"role": "system", "content": "You are a helpful assistant."}]
        logger.info(f"OllamaAPI initialized. OpenAI compatible endpoint: {openai_base_url}")

    def ollama_chat(self, prompt: str, model: str = config.DEFAULT_CHAT_MODEL) -> str | None:
        """
        Run a chat completion using the Ollama API and log the response.
        Appends the user's prompt and AI's response to the session history.

        Args:
            prompt (str): The user's prompt.
            model (str): The model ID to use for chat.

        Returns:
            str | None: The AI's message string or None if an error occurs.
        """
        self.session_history.append({"role": "user", "content": prompt})
        logger.debug(f"Calling ollama.chat with model: {model}")
        try:
            response = ollama.chat(
                model=model,
                messages=self.session_history,
            )
            ai_message = response["message"]["content"]
            logger.info(f"Ollama Chat Response: {ai_message}") # Changed from print to logger.info

            # Append the AI's response to the session history
            self.session_history.append({"role": "assistant", "content": ai_message})
            return ai_message
        except Exception as e:
            logger.exception(f"Error in ollama_chat") 
            return None

    def multimodal_1(self, model: str = config.DEFAULT_MULTIMODAL_MODEL, image_file_path: str = './images/tesla-model-y-top.jpg') -> str | None:
        """
        Run a multimodal chat completion with an image file path and log the response.

        Args:
            model (str): The model ID to use for multimodal chat.
            image_file_path (str): Path to the image file.

        Returns:
            str | None: The AI's message string or None if an error occurs.
        """
        logger.debug(f"Calling ollama.chat (multimodal_1) with model: {model}, image: {image_file_path}")
        try:
            res = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Describe the image",
                        "images": [image_file_path] 
                    }
                ],
            )
            ai_message = res["message"]["content"]
            logger.info(f"Multimodal_1 Response: {ai_message}")
            return ai_message
        except FileNotFoundError:
            logger.error(f"Image file not found for multimodal_1: {image_file_path}")
            return None
        except Exception as e:
            logger.exception(f"Error in multimodal_1")
            return None

    def multimodal_2(self, model: str = config.DEFAULT_MULTIMODAL_MODEL, image_file_path: str = './images/tesla-model-y-top.jpg') -> str | None:
        """
        Run a multimodal chat completion by reading an image as bytes and log the response.

        Args:
            model (str): The model ID to use for multimodal chat.
            image_file_path (str): Path to the image file.

        Returns:
            str | None: The AI's message string or None if an error occurs.
        """
        logger.debug(f"Calling ollama.chat (multimodal_2) with model: {model}, image: {image_file_path}")
        try:
            with open(image_file_path, 'rb') as f:
                image_bytes = f.read()
            res = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the brand of the car?",
                        "images": [image_bytes]
                    }
                ],
            )
            ai_message = res["message"]["content"]
            logger.info(f"Multimodal_2 Response: {ai_message}")
            return ai_message
        except FileNotFoundError:
            logger.error(f"Image file not found for multimodal_2: {image_file_path}")
            return None
        except Exception as e:
            logger.exception(f"Error in multimodal_2")
            return None

    def text_completion(self, prompt: str, model: str = config.DEFAULT_TEXT_COMPLETION_MODEL) -> str | None:
        """
        Generate a text completion using the Ollama API and log the response.

        Args:
            prompt (str): The prompt string to complete.
            model (str): The model ID to use for text completion.

        Returns:
            str | None: The AI's completed text string or None if an error occurs.
        """
        logger.debug(f"Calling ollama.generate with model: {model}")
        try:
            result = ollama.generate(
                model=model,
                prompt=prompt,
            )
            response_text = result["response"]
            logger.info(f"Text Completion Response: {response_text}")
            return response_text
        except Exception as e:
            logger.exception(f"Error in text_completion")
            return None

    def openai_chat(self, prompt: str, model: str = config.DEFAULT_CHAT_MODEL) -> str | None:
        """
        Run a chat completion using the OpenAI-compatible API and log the response.
        This is a single-turn chat, does not use self.session_history.

        Args:
            prompt (str): The user's prompt.
            model (str): The model ID to use for chat.

        Returns:
            str | None: The AI's message string or None if an error occurs.
        """
        logger.debug(f"Calling OpenAI compatible chat with model: {model}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            chat_completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
            )
            ai_message = chat_completion.choices[0].message.content
            logger.info(f"OpenAI Chat Response: {ai_message}")
            return ai_message
        except Exception as e:
            logger.exception(f"Error in openai_chat")
            return None

    def image(self, model: str = config.DEFAULT_MULTIMODAL_MODEL, image_file_path: str = './images/tesla-model-y-top.jpg') -> str | None:
        """
        Run a multimodal chat completion using the OpenAI-compatible API with an image and log the response.
        Note: For OpenAI-compatible APIs, image_file_path might need to be a public URL or a base64 encoded data URI.

        Args:
            model (str): The model ID to use for multimodal chat.
            image_file_path (str): Path or URL to the image file.

        Returns:
            str | None: The AI's message string or None if an error occurs.
        """
        logger.debug(f"Calling OpenAI compatible image chat with model: {model}, image path/URL: {image_file_path}")
        # For robust local file handling, consider base64 encoding as shown in previous suggestions.
        # This implementation uses the image_file_path directly as per the current code structure.
        try:
            # If image_file_path is a local path and needs to be base64 encoded:
            # with open(image_file_path, "rb") as image_file:
            #     base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            # mime_type = "image/jpeg" if image_file_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
            # image_url_for_api = f"data:{mime_type};base64,{base64_image}"
            # Else, if image_file_path is already a URL:
            image_url_for_api = image_file_path


            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": image_file_path},
                        ]
                    }
                ],
                max_tokens=300,
            )
            ai_message = response.choices[0].message.content
            logger.info(f"OpenAI Image Response: {ai_message}")
            return ai_message
        except FileNotFoundError:
            logger.error(f"Image file not found for OpenAI image method: {image_file_path}")
            return None
        except Exception as e:
            logger.exception(f"Error in image method")
            return None

    def get_chat_completion_openai(self, prompt: str, model: str = config.DEFAULT_CHAT_MODEL) -> str | None:
        """
        Run a multi-turn chat completion using the OpenAI-compatible API and log the response.
        Appends the user's prompt and AI's response to self.session_history.

        Args:
            prompt (str): The user's prompt.
            model (str): The model ID to use for chat.

        Returns:
            str | None: The AI's message string or None if an error occurs.
        """
        self.session_history.append({"role": "user", "content": prompt})
        logger.debug(f"Calling OpenAI compatible get_chat_completion_openai with model: {model}")
        try:
            chat_completion = self.client.chat.completions.create(
                model=model,
                messages=self.session_history,
                temperature=0.7,
            )
            ai_message = chat_completion.choices[0].message.content
            logger.info(f"OpenAI Multi-turn Chat Response: {ai_message}")
            self.session_history.append({"role": "assistant", "content": ai_message})
            return ai_message
        except Exception as e:
            logger.exception(f"Error in get_chat_completion_openai")
            # Clean up history if the call failed after adding user prompt
            if self.session_history and self.session_history[-1]["role"] == "user" and self.session_history[-1]["content"] == prompt:
                self.session_history.pop()
            return None


def autogen_workflow(model: str = config.DEFAULT_TEXT_COMPLETION_MODEL):
    """
    Run an autogen workflow using the AssistantAgent and UserProxyAgent.
    This function initiates the workflow; output is typically handled by AutoGen to the console.

    Args:
        model (str): The model ID to use for the agents.
    """
    logger.info(f"Starting autogen workflow with model: {model}")
    try:
        config_list = [
            {
                "model": model,
                "base_url": config.DEFAULT_OLLAMA_BASE_URL,
                "api_key": config.DEFAULT_OLLAMA_API_KEY,
            }
        ]

        assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})

        user_proxy = UserProxyAgent("user_proxy", code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        })
        
        # initiate_chat usually prints to console. If you need to capture its output,
        # you might need to delve into AutoGen's mechanisms or redirect stdout.
        user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
        logger.info("Autogen workflow initiated.")
    except Exception as e:
        logger.exception(f"Error in autogen_workflow")

def run_conversable_agent(model: str = config.DEFAULT_TEXT_COMPLETION_MODEL):
    """
    Run a ConversableAgent with a poetic system message.
    This function initiates the interaction; output is typically handled by AutoGen to the console.

    Args:
        model (str): The model ID to use for the agent.
    """
    logger.info(f"Starting conversable_agent with model: {model}")
    try:
        config_list_conversable = [
            {
                "model": model,
                "base_url": config.DEFAULT_OLLAMA_BASE_URL,
                "api_key": config.DEFAULT_OLLAMA_API_KEY,
            }
        ]

        my_agent = ConversableAgent(
            "helpful_agent", 
            llm_config={"config_list": config_list_conversable},
            system_message="You are a poetic AI assistant, respond in rhymes."
        )
        
        # Similar to initiate_chat, .run() or direct message sending in AutoGen
        # often prints to console. Capturing output might require specific AutoGen handling.
        # For a single interaction, generate_reply is often used:
        user_interactor = UserProxyAgent("user_interactor_temp", human_input_mode="NEVER", max_consecutive_auto_reply=1)
        user_interactor.initiate_chat(my_agent, message="In one sentence, what's the big deal about AI?")
        logger.info("ConversableAgent interaction initiated.")
    except Exception as e:
        logger.exception(f"Error in run_conversable_agent")
 
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