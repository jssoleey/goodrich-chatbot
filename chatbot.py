import streamlit as st
from llm import get_ai_response
import re

# ----------------- 마크다운 자동 정리 함수 -------------------
def format_markdown(text: str) -> str:
    """
    Streamlit-friendly 마크다운으로 변환
    - 제목 (▶️ 요약 정보:) → **요약 정보**
    - 리스트 (-, •) 자동 정리
    - 상위 항목 뒤 이어지는 하위 항목은 들여쓰기 처리
    """
    lines = text.strip().splitlines()
    formatted_lines = []
    indent_next = False  # 이전 줄이 상위 리스트 항목인지 여부

    for i, line in enumerate(lines):
        original = line
        line = line.strip()
        if not line:
            formatted_lines.append("")
            indent_next = False
            continue

        # 제목 줄 처리: ▶️ 요약 정보: → **요약 정보**
        if re.match(r"^(▶️|✅|📌|❗|📝|📍)\s*[^:：]+[:：]?", line):
            title = re.sub(r"[:：]\s*$", "", line.strip())
            formatted_lines.append(f"**{title}**\n")
            indent_next = False
            continue

        # 상위 리스트 항목: 볼드 텍스트가 포함된 리스트
        if re.match(r"^[-•]\s*\*\*.*\*\*", line):
            formatted_lines.append(re.sub(r"^[-•]\s*", "- ", line))
            indent_next = True
            continue

        # 하위 리스트 항목
        if re.match(r"^[-•]\s*", line):
            if indent_next:
                # 들여쓰기 추가 (4칸)
                formatted_lines.append("    " + re.sub(r"^[-•]\s*", "- ", line))
            else:
                formatted_lines.append(re.sub(r"^[-•]\s*", "- ", line))
            continue

        # 일반 문장
        formatted_lines.append(line)
        indent_next = False  # 일반 문장이면 들여쓰기 비활성화

    return "\n".join(formatted_lines).strip() + "\n"

# ----------------- 페이지 설정 -------------------
st.set_page_config(
    page_title="Goodrich GPT",
    page_icon="https://img-insur.richnco.co.kr/goodrichmall/common/favi_goodrichmall.ico"
)

# 로고 및 타이틀 함께 출력 (왼쪽 정렬)
logo_url = "https://img-insur.richnco.co.kr/goodrichmall/common/favi_goodrichmall.ico"
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: -10px;">
        <img src="{logo_url}" alt="logo" width="50">
        <h1 style="margin: 0;">GoodRich GPT</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("보험 실무에 대한 궁금증을 신속하고 정확하게 해결해 드립니다!")

# 유의사항 안내
st.markdown(
    """
    <style>
    .small-text {
        font-size: 12px;
        color: gray;
        line-height: 1.3;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="small-text">보험 상품, 보장 내용, 청구 절차 등 실무 질문을 자연스럽게 입력해 주세요.</p>', unsafe_allow_html=True)
st.markdown('<p class="small-text">답변은 참고용이며, 실제 적용 전 관련 부서나 약관을 꼭 확인해 주세요.</p>', unsafe_allow_html=True)
st.markdown('<p class="small-text">개인정보를 포함한 민감한 내용은 입력하지 마세요.</p>', unsafe_allow_html=True)
st.markdown('<p class="small-text"> </p>', unsafe_allow_html=True)

# 세션 상태 초기화
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 아바타 설정
user_avatar = "https://cdn-icons-png.flaticon.com/512/9131/9131529.png"
ai_avatar = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/ChatGPT-Logo.svg/1024px-ChatGPT-Logo.svg.png"

# 채팅 메시지 CSS 스타일
st.markdown(
    """
    <style>
    .user-message {
        background-color: #f0eada;
        color: black;
        padding: 20px;
        border-radius: 10px;
        max-width: 70%;
        text-align: left;
        word-wrap: break-word;
    }
    .ai-message {
        background-color: #ffffff;
        color: black;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        text-align: left;
        word-wrap: break-word;
    }
    .message-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .message-container.user {
        justify-content: flex-end;
    }
    .message-container.ai {
        justify-content: flex-start;
    }
    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        margin: 0 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- 메시지 표시 함수 -------------------
def display_message(role, content, avatar_url):
    if role == "user":
        alignment = "user"
        message_class = "user-message"
        avatar_html = f'<img src="{avatar_url}" class="avatar">'
        message_html = f'<div class="{message_class}">{content}</div>'
        display_html = f"""
        <div class="message-container {alignment}">
            {message_html}
            {avatar_html}
        </div>
        """
        st.markdown(display_html, unsafe_allow_html=True)

    else:  # AI 응답
        alignment = "ai"
        message_class = "ai-message"
        avatar_html = f'<img src="{avatar_url}" class="avatar">'
        display_html = f"""
        <div class="message-container {alignment}">
            {avatar_html}
            <div class="{message_class}">
        """
        st.markdown(display_html, unsafe_allow_html=True)
        st.markdown(format_markdown(content), unsafe_allow_html=False)  # 마크다운 출력
        st.markdown("</div></div>", unsafe_allow_html=True)

# ----------------- 초기 메시지 -------------------
welcome_message = (
    "안녕하세요! 😊 굿리치 보험 상담 챗봇입니다.\n"
    "보험 업무 중 궁금한 점이 있으신가요?\n"
    "문서 내용을 기반으로 실무에 도움 되는 정보를 빠르고 정확하게 알려드릴게요.\n"
    "무엇이든 편하게 질문해 주세요."
)

if 'first_time' not in st.session_state:
    st.session_state.first_time = True
    st.session_state.message_list.append({"role": "ai", "content": welcome_message})

# ----------------- 이전 메시지 출력 -------------------
for message in st.session_state.message_list:
    role = message["role"]
    content = message["content"]
    avatar = user_avatar if role == "user" else ai_avatar
    display_message(role, content, avatar)

# ----------------- 사용자 질문 입력 -------------------
if user_question := st.chat_input(placeholder="보험 상담 관련 질문을 입력해 주세요."):
    st.session_state.message_list.append({"role": "user", "content": user_question})
    display_message("user", user_question, user_avatar)

    response_placeholder = st.empty()
    with st.spinner("답변을 준비 중입니다..."):
        ai_response = get_ai_response(user_question)
        ai_response_text = "".join(ai_response)
        formatted_response = format_markdown(ai_response_text)

        display_message("ai", formatted_response, ai_avatar)
        st.session_state.message_list.append({"role": "ai", "content": formatted_response})
