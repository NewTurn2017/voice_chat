import streamlit as st
from utils import print_messages, StreamHandler, generate_summary, text_to_speech, autoplay_audio
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os
import uuid
import requests

# 스트림릿 앱의 기본 설정
st.set_page_config(page_title="Voice Chat App", page_icon="🤖")
st.title('티처블 러신머닝 기반🤔 보이스챗🎙️')

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "store" not in st.session_state:
    st.session_state["store"] = {}
if "chat_ids" not in st.session_state:
    st.session_state["chat_ids"] = []
if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}
if "llm" not in st.session_state:
    st.session_state["llm"] = ChatOpenAI(streaming=True, callbacks=[
                                         StreamHandler(st.empty())], max_tokens=50, temperature=0.3)
if "prompt" not in st.session_state:
    st.session_state["prompt"] = ChatPromptTemplate.from_messages([
        ("system", "You recognise the person in the photo you uploaded as the person you're talking to, and you always call them by their first name, just like a friend."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


def start_new_chat():
    if st.session_state["messages"]:
        summary = generate_summary(st.session_state["messages"])
        save_chat_history(summary)
    st.session_state["messages"] = []


def save_chat_history(summary=None):
    if not st.session_state["messages"]:
        st.warning("저장할 대화가 없습니다.")
        return
    chat_id = str(uuid.uuid4())
    title = summary if summary else f"대화 {chat_id[:8]}"
    st.session_state["chat_titles"][chat_id] = title
    st.session_state["store"][chat_id] = st.session_state["messages"]
    st.session_state["chat_ids"].append(chat_id)
    st.session_state["messages"] = []


def load_chat_history(chat_id):
    if chat_id in st.session_state["store"]:
        st.session_state["messages"] = st.session_state["store"][chat_id]


def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = StreamlitChatMessageHistory()
    return st.session_state["store"][session_ids]


# 사이드바 설정
with st.sidebar:
    if st.sidebar.button("새 대화 시작"):
        start_new_chat()
    chat_history_list = st.selectbox(
        "대화 목록",
        options=st.session_state["chat_ids"],
        format_func=lambda x: st.session_state["chat_titles"][x],
        key="chat_history_list"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("대화 불러오기"):
            load_chat_history(chat_history_list)
    with col2:
        if st.button("대화 저장"):
            if st.session_state["messages"]:
                summary = generate_summary(st.session_state["messages"])
                save_chat_history(summary)
            else:
                st.warning("저장할 대화가 없습니다.")

    tts_models = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    tts_model = st.selectbox("음성선택", tts_models)
    selected_speed = st.slider(
        "말하기 속도", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    st.sidebar.markdown("---")

    st.sidebar.header("사진 촬영")
    use_webcam = st.checkbox("웹캠 사용")

    if use_webcam:
        picture = st.camera_input("웹캠을 사용하여 사진을 찍으세요")
        # 웹캠을 사용하여 사진을 찍었을 때의 처리 추가
        if use_webcam and picture:
            st.image(picture)
            files = {"file": ("webcam_image.png",
                              picture.getvalue(), "image/png")}
            response = requests.post(
                "http://127.0.0.1:8000/predict/", files=files)
            if response.status_code == 200:
                result = response.json()
                st.write(f"Class: {result['class']}")
                st.write(f"Confidence Score: {result['confidence_score']}")
                chain_with_memory = RunnableWithMessageHistory(
                    st.session_state["prompt"] | st.session_state["llm"], get_session_history, input_messages_key="question", history_messages_key="history")
                response = chain_with_memory.invoke({"question": f"사진을 업로드했습니다: {result['class']}"}, config={
                                                    "configurable": {"session_id": str(uuid.uuid4())}})
                st.session_state["messages"].append(ChatMessage(
                    role="assistant", content=response.content))
                st.chat_message("assistant").write(response.content)
                speech_file_path = text_to_speech(
                    response.content, voice=tts_model, speed=selected_speed)
                autoplay_audio(speech_file_path)
            else:
                st.write("Error: Unable to get prediction")

    st.sidebar.header("이미지 업로드")
    uploaded_image = st.file_uploader(
        "이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image.',
                 use_column_width=True)
        files = {"file": (uploaded_image.name,
                          uploaded_image.getvalue(), uploaded_image.type)}
        response = requests.post("http://127.0.0.1:8000/predict/", files=files)
        if response.status_code == 200:
            result = response.json()
            st.write(f"Class: {result['class']}")
            st.write(f"Confidence Score: {result['confidence_score']}")
            chain_with_memory = RunnableWithMessageHistory(
                st.session_state["prompt"] | st.session_state["llm"], get_session_history, input_messages_key="question", history_messages_key="history")
            response = chain_with_memory.invoke({"question": f"사진을 업로드했습니다: {result['class']}"}, config={
                                                "configurable": {"session_id": str(uuid.uuid4())}})
            st.session_state["messages"].append(ChatMessage(
                role="assistant", content=response.content))
            st.chat_message("assistant").write(response.content)
            speech_file_path = text_to_speech(
                response.content, voice=tts_model, speed=selected_speed)
            autoplay_audio(speech_file_path)
        else:
            st.write("Error: Unable to get prediction")

# 이전 대화기록 출력
print_messages()

# 사용자 입력 처리
if user_input := st.chat_input("메시지를 입력하세요"):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(
        ChatMessage(role="user", content=user_input))
    chain_with_memory = RunnableWithMessageHistory(
        st.session_state["prompt"] | st.session_state["llm"], get_session_history, input_messages_key="question", history_messages_key="history")
    response = chain_with_memory.invoke({"question": user_input}, config={
                                        "configurable": {"session_id": str(uuid.uuid4())}})
    st.session_state["messages"].append(ChatMessage(
        role="assistant", content=response.content))
    st.chat_message("assistant").write(response.content)
    speech_file_path = text_to_speech(
        response.content, voice=tts_model, speed=selected_speed)
    autoplay_audio(speech_file_path)
