Playing around with both LM Studio and Ollama.
For sake of simplicity, not touching llama.cpp

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
