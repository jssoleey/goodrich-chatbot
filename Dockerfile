# 1. 파이썬 기반 이미지 사용
FROM python:3.10

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 소스 코드 전체 복사
COPY . /app

# 4. 필요 패키지 설치
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5. 웹 포트 열기
EXPOSE 8501

# 6. Streamlit 실행
CMD ["streamlit", "run", "chatbot.py", "--server.port=8501", "--server.enableCORS=false"]
