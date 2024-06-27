import streamlit as st
from PIL import Image
from aisam_utils import print_messages, StreamHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import openai
from openai import OpenAI
import os

import whisper
import sounddevice as sd
import wavio
import time
import random  #수강생 id의 value를 random으로 받을 수 있게 함

# Streamlit 앱의 첫 번째 명령으로 set_page_config()를 한 번만 호출해야 합니다.
# 이미 다른 곳에서 set_page_config()를 호출했다면, 그 부분을 제거하거나 이 코드를 그 부분으로 옮겨주세요.
# 페이지 설정
st.set_page_config(layout="wide")

# 이미지 로드
image = Image.open('heeja3.png')

# 이미지와 제목 표시를 위한 컬럼 생성
col1, col2 = st.columns([1, 3])  # 비율을 조정하여 이미지와 텍스트의 레이아웃을 조절할 수 있습니다.
with col1:
    st.image(image, width=80)
with col2:
    st.title("👨‍🏫성찰과 문해력, ai훈장샘에서 찾다👨‍🏫")
# 확장 가능한 섹션 추가
with st.expander("ai훈장샘은요"):
    st.markdown("# 너는 여전히 논어를 번역서로 보니?")
    st.markdown("# 난 원문으로 읽는다!")
    #st.set_page_config(page_title="ai훈장샘", page_icon="👨‍🏫")
    # st.subheader("👨‍🏫 성찰과 문해력, 고전에서 찾다 🤖")
    st.markdown('###🤖ai훈장샘은 고전을 보다 쉽고 즐겁게 배울 수 있게 지원합니다.')

# Whisper 모델 로드
model = whisper.load_model("base")

# 로컬 오디오 파일 경로 지정
#audio_file = "introduction.mp3"
video_file = "heeja_video.mp4"

# 인사 버튼 클릭 시 mp3 파일 재생
if st.button('인사'):
    video_file = os.path.join(os.getcwd(), 'heeja_video.mp4')  # 파일 경로 설정
    video_bytes = open(video_file, 'rb').read()
    st.video(video_bytes, format='video/mp4')

# API 초기화
def init_api():
    with open("chatgpt.env") as env:
      for line in env:
        key, value = line.strip().split("=")
        os.environ[key] = value

init_api()

# OpenAI API 키 설정
client = OpenAI(api_key=os.environ.get("API_KEY"))

# 오디오 녹음 설정
duration = 10  # 초
fs = 44100  # 샘플 레이트

# 파일 경로
audio_file2 = "recording.mp3"

# 오디오를 녹음하는 함수
def record_audio(duration, fs, file_path):
    st.info("녹음 중...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)

    # 녹음 진행 상황을 표시
    progress_bar = st.progress(0)
    for i in range(duration):
        time.sleep(1)
        progress_bar.progress((i + 1) / duration)
    
    sd.wait()  # 녹음이 끝날 때까지 대기
    wavio.write(file_path, recording, fs, sampwidth=2)
    st.success("녹음 완료!")

# 계선편 버튼 클릭 시 오디오 녹음 및 확인 메시지 출력
if st.button('계선편을 읽어주세요.'):
    record_audio(duration, fs, audio_file2)
    st.write("잘 읽으셨습니다.")

# Whisper를 사용하여 오디오를 문자로 변환하는 함수
def transcribe_audio(file_path):
    st.info("변환 중...")
    result = model.transcribe(file_path)
    st.success("변환 완료!")
    return result["text"]

if st.button("학습자 읽은 내용"):
    transcription = transcribe_audio(audio_file2)
    st.text_area("음독 결과", transcription)

if st.button("버튼을 누르면 한자를 볼 수 있어요"):
    st.write('子曰, 爲善子는 天報之以福하고 爲不善子는 天報之以禍니라, 입니다.')

button = st.button('해석입니다.')
if button:
    st.markdown('''
        子曰, 爲善子는 天報之以福하고 爲不善子는 天報之以禍니라
        :blue[공자께서 말씀하시길, 
        선한 일을 하는 사람은 하늘이 복으로써 답하고 
        선을 행하지 않는 자는 하늘이 재앙으로써 답하니라.]
        ''')

# AI 대화 기능
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 채팅 대화기록을 저장하는 store 세션 상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()

with st.sidebar:
    session_id = st.text_input("수강생 ID", value=str(random.randint(1,10000)))  # 대화방의 아이디

    clear_btn = st.button("수강생 학습기록 초기화")
    if clear_btn:
        st.session_state["messages"] = []
        # st.session_state["store"] = dict()  대화기록 초기화
        st.experimental_rerun()

# 이전 대화기록을 출력해 주는 코드
print_messages()

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    # 주어진 user_id와 conversation_id에 해당하는 세션 기록을 반환합니다.
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

# 사용자 입력 처리
if user_input := st.chat_input("질문해 주세요"):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    
    # AI의 답변 생성
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        # LLM을 사용하여 AI의 답변을 생성
        llm = ChatOpenAI(openai_api_key=os.environ.get("API_KEY"), 
                         streaming=True, callbacks=[stream_handler])

        # 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "너는 한문을 많이 알고 있는 동양철학 교수야. 질문에 간결하면서도 따뜻하게 답변해 주세요"),
                MessagesPlaceholder(variable_name="history"),  # 대화 기록을 변수로 사용
                ("human", "{question}"),  # 사용자의 질문을 입력
            ]
        )

        chain = prompt | llm

        chain_with_memory = RunnableWithMessageHistory(
            chain,  # 실행할 Runnable 객체
            get_session_history,  # 세션 기록을 가져오는 함수
            input_messages_key="question",  # 사용자 질문의 키
            history_messages_key="history",  # 기록 메시지의 키
        )

        response = chain_with_memory.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}},  # 세션ID 설정
        )

    st.session_state["messages"].append(
        ChatMessage(role="assistant", content=response.content)
    )
