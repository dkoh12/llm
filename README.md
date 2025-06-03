# Local LLM

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Ollama-0.1.34%2B-green?logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjMDAwMDAwIiBoZWlnaHQ9IjE2IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1zbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMiIgZmlsbD0iIzAwZDY2ZiIvPjwvc3ZnPg==" alt="Ollama 0.1.34+">
  <img src="https://img.shields.io/badge/LM%20Studio-0.2.20%2B-purple" alt="LM Studio 0.2.20+">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
</p>


Playing around with both LM Studio and Ollama.
For sake of simplicity, not touching llama.cpp

Vibecoded with Github Copilot - GPT-4.1, Gemini 2.5 Pro, Claude Opus 4

## Setup

1. clone this repo

2. Install `uv` - fast python package manager. This is optional as pip is fine too.
`pip install uv`

3. set up virtual env (recommended)

```
python -m venv venv
source venv/bin/activate    # on windows, venv\Scripts\activate
```

or you can use `uv`
```
uv venv
source venv/bin/activate    # on windows, venv\Scripts\activate
```

4. install dependencies

```
pip install -r requirements.txt
```

or
```
uv pip install -r requirements.txt
```

## Playing with LM Studio

1. download LM studio (https://lmstudio.ai/)

2. download local LLM models from LM Studio
   
run
```
python lmstudio_api.py
```

## Playing with Ollama

1. download Ollama (https://ollama.com/)

2. download local LLM models from Ollama

3. run ollama server

```
ollama run codellama:latest
```

run
```
python ollama_api.py
```

Note Ollama and LM Studio downloads models in different locations. There are OSS tools out there to create a symlink out there.
This repo does not come with that. 

## Playing with LLM

To play with the overall LLM Agent / chatbot, run

```
python llm.py
```

Available Commands
- Chat
  - chat
- Completion
  - auto complete
- Models
  - To list all models from the selected provider (Ollama / LMStudio)
- Select
  - To select a model
- Switch
  - To switch between Ollama and LMStudio

Note - inference is kind of slow despite running the model on localhost

## Running Tests

To run the unit tests for the API wrappers and agent, use the following commands from the root of the project:

To run all tests in the `tests` directory:
```
python -m unittest discover tests
```

To run a specific test file, for example for LMStudioAPI:
```
python -m unittest tests/test_lmstudio_api.py
```

Make sure your LM Studio and/or Ollama servers are running before running the tests, as the tests will attempt to connect to them.


## Linting

Linting is done with ruff. It's not yet hooked up to git hook.

```
ruff check . --fix
```

```
ruff format .
```

## Project Management

Project Management is done through `pyproject.toml`

```
# For main dependencies
uv pip compile pyproject.toml -o requirements.txt

# For development dependencies
uv pip compile pyproject.toml --extra dev -o requirements-dev.txt
```

```
# For development
uv pip sync requirements-dev.txt

# For production/CI (if you don't have dev tools there)
uv pip sync requirements.txt
```

---

## License

This project is licensed under the [MIT License](LICENSE).
