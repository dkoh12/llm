Playing around with both LM Studio and Ollama.
For sake of simplicity, not touching llama.cpp

Vibecoded with Github Copilot - GPT4.1

## playing with LM Studio

1. clone this repo

2. set up vitrual env (recommended)

```
python -m venv venv
source venv/bin/activate    # on windows, venv\Scripts\activate
```

3. install dependencies

```
pip install -r requirements.txt
```

4. download LM studio (https://lmstudio.ai/)

run
```
python lmstudio_api.py
```

## Playing with Ollama

5. download Ollama (https://ollama.com/)

6. run ollama server

```
ollama run codellama:latest
```

run
```
python ollama_api.py
```

Search up any python packages on PyPi
https://pypi.org/project/pyautogen/#description

## Playing with LLM

To play with the overall LLM Agent / chatbot, run

```
python llm.py
```

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

---
