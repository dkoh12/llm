# Default configuration settings for OllamaAPI

# API Endpoint Configuration
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_API_KEY = "ollama" # Often "ollama" or can be an empty string if not required

# Default Model Names
DEFAULT_CHAT_MODEL = "llama3.2"
DEFAULT_MULTIMODAL_MODEL = "llava:7b"
DEFAULT_TEXT_COMPLETION_MODEL = "codellama:latest"

# Image paths (if you want to configure them)
# DEFAULT_IMAGE_PATH = "./images/tesla-model-y-top.jpg"

# LM Studio Configuration
DEFAULT_LMSTUDIO_SERVER = "http://localhost:1234"
# LM Studio typically doesn't require an API key for local instances,
# but we can have a placeholder. The OpenAI client for LM Studio uses "lm-studio" by default.
DEFAULT_LMSTUDIO_API_KEY = "lm-studio" 

# Default Model Names for LM Studio (these are examples, adjust to your loaded models)
DEFAULT_LMSTUDIO_CHAT_MODEL = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF" # Example
DEFAULT_LMSTUDIO_COMPLETION_MODEL = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF" # Example
DEFAULT_LMSTUDIO_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF" # Example