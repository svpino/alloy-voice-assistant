import base64

import openai
import pyaudio
import speech_recognition as sr
import whisper
from cv2 import VideoCapture, destroyWindow, imencode, imshow, waitKey
from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

CAM_PORT = 0

current_frame = None

whisper_model = whisper.load_model("small")
model = ChatOpenAI(model="gpt-4o")

SYSTEM_PROMPT = """
You are a witty assistant that will use the image provided by the user
to answer its questions.

Use few words on your questions. Go straight to the point. Do not use any
emoticons or emojis. Do not ask follow-up questions.

You have memory, so you can use the entire context of the conversation to
tailor your answers. You can also use the image provided by the user.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            [
                {"type": "text", "text": "{question}"},
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_base64}",
                },
            ],
        ),
    ]
)

parser = StrOutputParser()
chain = prompt_template | model | parser

chat_message_history = ChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_message_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


def inference(text, image):
    print("Sending request to the model...")

    response = chain_with_history.invoke(
        {"question": text, "image_base64": image.decode()},
        config={"configurable": {"session_id": "unused"}},
    )

    return response


def tts(text):
    player_stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16, channels=1, rate=24000, output=True
    )

    with openai.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        response_format="pcm",
        input=text,
    ) as response:
        for chunk in response.iter_bytes(chunk_size=1024):
            player_stream.write(chunk)


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, language="english")

        if prompt:
            print("Prompt:", prompt)

            _, buffer = imencode(".jpeg", current_frame)
            encoded_image = base64.b64encode(buffer)

            result = inference(prompt, encoded_image)
            print(result)

            tts(result)

    except sr.UnknownValueError:
        print("We couldn't understand audio")


r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(source)

stop_listening = r.listen_in_background(m, audio_callback)

# calling this function requests that the background listener stop listening
# stop_listening(wait_for_stop=False)

while True:
    cam = VideoCapture(CAM_PORT)
    result, current_frame = cam.read()

    if result:
        imshow("webcam", current_frame)

    if waitKey(1) & 0xFF == ord("q"):
        break

destroyWindow("webcam")
