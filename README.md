Playing around with both LM Studio and Ollama.
For sake of simplicity, not touching llama.cpp

## playing with LM Studio

1. clone this repo

2. download LM studio

3. set up vitrual env (recommended)

```
python -m venv venv
source venv/bin/activate    # on windows, venv\Scripts\activate
```

4. install dependencies

```
pip install -r requirements.txt
```

run
```
python lmstudio_api.py
```

## Playing with Ollama

5. run ollama server

```
ollama run codellama:latest
```

run
```
python ollama_api.py
```


Search up any python packages on PyPi
https://pypi.org/project/pyautogen/#description


