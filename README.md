# Sample AI assistant

You need an `OPENAI_API_KEY` and a `GOOGLE_API_KEY` to run this code. Store them in a `.env` file in the root directory of the project, or set them as environment variables.


If you are running the code on Apple Silicon, run the following command:

```
$ brew install portaudio
```

Create a virtual environment, update pip, and install the required packages:

```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```

Run the assistant:

```
$ python3 assistant.py
```