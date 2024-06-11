from langchain_core.callbacks.base import BaseCallbackHandler
import streamlit as st
import openai
from pathlib import Path
import base64


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def print_messages():
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(f"{chat_message.content}")


def generate_summary(messages):
    # 시스템 메시지를 추가하여 대화의 컨텍스트를 설정합니다.
    system_message = {
        "role": "system",
        "content":  "이 대화의 주요 내용을 간략하게 요약해주세요."
    }
    # 사용자와 어시스턴트의 메시지를 포맷에 맞게 변환합니다.
    formatted_messages = [
        {"role": msg.role, "content": msg.content} for msg in messages]
    # 시스템 메시지를 대화 목록의 시작 부분에 추가합니다.
    messages_with_context = [system_message] + formatted_messages
    print(messages_with_context)
    # OpenAI API를 사용하여 요약 생성
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_with_context,
        max_tokens=30,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0,
        stop=["\n"]
    )

    # 생성된 텍스트를 반환합니다.
    # 응답에서 마지막 메시지(요약)의 내용을 추출합니다.
    last_message_content = response.choices[0].message.content.strip()
    return last_message_content


def text_to_speech(text, voice="alloy", response_format="mp3", speed=1):
    # 음성 파일을 저장할 경로 설정
    speech_file_path = Path(__file__).parent / "speech.mp3"

    # OpenAI API를 사용하여 음성 생성 요청
    response = openai.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format=response_format,
        speed=speed,
    )

    response.write_to_file(speech_file_path)

    return speech_file_path


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )
