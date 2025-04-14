from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, PrivateAttr
from typing import List
from functools import lru_cache
import streamlit as st
import os
from dotenv import load_dotenv
import traceback

# ======================== 설정 ========================
load_dotenv(dotenv_path=".envfile", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
EMBEDDING_MODEL = "solar-embedding-1-large"
PERSIST_DIR = "./chroma_rag_demo"
COLLECTION_NAME = "rag-demo"

# ======================== 전역 저장소 ========================
store = {}

# ======================== 전역 프롬프트 ========================
SYSTEM_PROMPT = (
    "당신은 보험 상담사의 역할을 수행하는 AI입니다.\n\n"
    "아래는 보험 상담사가 고객님과 전화 상담을 진행할 때 참고하는 문서입니다. "
    "이 문서를 바탕으로, 상담원이 고객님께 직접 전화로 안내하듯이 자연스럽고 친절한 말투로 응답해 주세요.\n\n"
    "[문서 요약 정보]\n{context}\n\n"
    "[고객 질문]\n{input}\n\n"
    "[응답 구성 지침]\n"
    "1. 답변은 다음 두 가지 형식으로 모두 구성해 주세요: 두 형식 사이에는 구분선을 넣어주세요.\n"
    "   - 첫째, 상담원이 고객님과 통화하며 그대로 읽을 수 있는 **스크립트 형식**으로 작성하세요.\n"
    "   - 둘째, 고객이 화면상에서 쉽게 이해할 수 있도록 **항목별 정리된 정보 블록 (마크다운)**으로 작성하세요.\n"
    "2. 스크립트는 ‘안녕하세요 고객님~’, ‘도움이 되셨으면 좋겠습니다~’ 같은 자연스러운 말투로 말해주세요.\n"
    "3. 정보 블록은 소제목, 목록, 강조, 줄바꿈 등을 적절히 사용해 보기 쉽게 구성해 주세요.\n"
    "4. 소제목은 너무 크지 않게 굵은 글씨나 박스 형태로, Streamlit에서도 잘 보이도록 작성해 주세요.\n"
    "5. 긴 설명은 나열하지 말고, 항목별로 요약 정리해 주세요.\n"
    "6. 문서에서 확인할 수 없는 내용은 확신하지 말고, 추가 확인이 필요하다고 안내해 주세요.\n"
    "7. 모든 응답은 반드시 상담용 구어체로 작성해 주세요.\n\n"
    "[응답 예시 형식]\n"
    "▶️ 요약 정보:\n"
    "마크다운 목록 형식으로 알아보기 쉽게 정리"
    "▶️ 상담 스크립트:\n"
    "안녕하세요, 고객님! 😊 문의하신 다자녀 할인 혜택에 대해 안내드릴게요. ...\n\n"
)

# ======================== 캐시된 함수 ========================
@lru_cache(maxsize=1)
def get_llm(model='gpt-4o-mini'):
    return ChatOpenAI(model=model)

@st.cache_resource
def get_vectorstore():
    embedding = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model=EMBEDDING_MODEL)
    return Chroma(collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIR, embedding_function=embedding)

@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ======================== Re-ranking 정의 ========================
def rerank_documents(query, docs, top_k=5):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = get_reranker().predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_k]]

class RerankRetriever(BaseRetriever):
    def __init__(self, base_retriever, top_k=5, **kwargs):
        super().__init__(**kwargs)
        self._base_retriever = base_retriever
        self._top_k = top_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self._base_retriever.get_relevant_documents(query)
        return rerank_documents(query, docs, top_k=self._top_k)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        docs = await self._base_retriever.aget_relevant_documents(query)
        return rerank_documents(query, docs, top_k=self._top_k)

# ======================== 세션 관리 ========================
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ======================== 체인 구성 ========================
def get_dictionary_chain():
    dictionary = ["BEFORE -> AFTER"]
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}

        질문: {{input}}
    """)
    return prompt | get_llm() | StrOutputParser()

def get_rerank_retriever():
    base_retriever = get_vectorstore().as_retriever(search_kwargs={"k": 10})
    return RerankRetriever(base_retriever=base_retriever, top_k=5)

def get_history_retriever():
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return create_history_aware_retriever(get_llm(), get_rerank_retriever(), contextualize_q_prompt)

def get_rag_chain():
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    retriever = get_history_retriever()
    rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(get_llm(), qa_prompt))

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

def get_ai_response(user_message, session_id="abc123"):
    try:
        dictionary_chain = get_dictionary_chain()
        rag_chain = get_rag_chain()
        rag_pipeline = {"input": dictionary_chain} | rag_chain

        result = rag_pipeline.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        )
        return iter([result])
    except Exception as e:
        import streamlit as st
        st.error("🔥 invoke 중 예외 발생! 콘솔 로그도 확인해주세요.")
        print("🔥 invoke 실행 중 예외 발생:", e)
        traceback.print_exc()  # 🔥 스택 트레이스 전체 출력
        return iter(["❌ 오류가 발생했습니다. 관리자에게 문의해 주세요."])

