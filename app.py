import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.documents import Document

load_dotenv()

st.set_page_config(page_title="2025 연말정산 상담 챗봇", page_icon="💰")

st.title("💰 2025년 연말정산 상담 챗봇")
st.markdown("""
2024년 귀속 연말정산 신고안내 및 2025년 개정세법을 기반으로 답변해드립니다.\n
**주의:** 정확한 세무 상담은 전문가와 상의하세요.
""")

# Initialize Chat Chain
@st.cache_resource
def get_chain():
    # 1. 벡터 저장소 설정
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    # k값을 10으로 늘려 '2025년에 바뀐 점' 같은 포괄적인 질문에 대해 충분한 문맥을 확보합니다.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 2. LLM 설정
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    # 3. 질문 맥락화 (대화 기록 반영)
    # 이 체인은 대화 기록을 바탕으로 사용자의 질문을 재구성하여 독립적인 질문으로 만듭니다.
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 4. 질문 답변 체인 (Stuff Documents Chain)
    # 이 체인은 문서와 질문을 받아 답변을 생성합니다.
    qa_system_prompt = """당신은 한국 연말정산 및 세법 전문가입니다. 
    아래 제공된 [문맥(Context)]을 바탕으로 질문에 답변하세요.

    [문맥(Context)]:
    {context}
    
    [답변 규칙]:
    1. **최우선 순위**: '2025년 개정세법' 또는 '2025년 귀속'과 관련된 내용이 문맥에 있다면, 이를 2024년 자료보다 우선하여 자세히 설명하세요.
    2. 질문이 '개정된 내용'이나 '달라진 점'을 묻는다면, 문맥에서 '개정', '신설', '확대', '인상' 등의 키워드가 포함된 내용을 종합하여 정리해 주세요.
    3. 문맥에 정답이 없다면 솔직하게 "제공된 자료에서 해당 내용을 찾을 수 없습니다."라고 답하세요. (단, 2025년 개정 내용이 조금이라도 보이면 최대한 활용하세요.)
    4. 답변은 친절하고 전문적인 한국어로 작성하세요.
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 5. 최종 검색 체인
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# 대화 기록을 위한 세션 상태
if "messages" not in st.session_state:
    st.session_state.messages = []

if "store" not in st.session_state:
    st.session_state.store = {}

session_id = "user_session"
if session_id not in st.session_state.store:
    st.session_state.store[session_id] = ChatMessageHistory()

# Streamlit이 다시 실행될 때 기록을 수동으로 관리하기 위한 헬퍼 함수
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state.store[session_id]

chain = get_chain()

# 대화 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("연말정산에 대해 궁금한 점을 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성 중입니다..."):
            
            # 현재 실행을 위해 Streamlit 기록을 LangChain 기록 형식으로 변환
            current_history = get_session_history(session_id)
            # (선택 사항: 필요한 경우 세션 상태 메시지를 히스토리 객체에 동기화하지만, 
            # 보통 RunnableWithMessageHistory가 지속성을 처리합니다. 
            # 여기서는 단순히 수동으로 기록을 전달하거나 업데이트합니다.)
            
            # UI를 위해 session_state에서 기록을 수동으로 관리하므로, 
            # 체인이 올바른 기록 형식을 받도록 해야 합니다.
            # 하지만 create_retrieval_chain은 RunnableWithMessageHistory 래퍼를 사용하지 않는 경우 
            # 입력에 'chat_history'가 필요합니다.
            # invoke를 위해 session_state에서 chat_history를 수동으로 구성합니다.
            
            from langchain_core.messages import HumanMessage, AIMessage
            chat_history = []
            for msg in st.session_state.messages[:-1]: # 현재 프롬프트 제외
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            # RAG Chain Invoke
            # "2025"와 "개정"이 포함된 질문인 경우, 2025년 전체 데이터를 문맥에 추가하여 요약 답변 유도
            if "2025" in prompt and ("개정" in prompt or "달라진" in prompt or "변화" in prompt):
                with open("data/2025년 개정세법.txt", "r", encoding="utf-8") as f:
                    full_2025_text = f.read()
                
                # 별도의 RAG 체인을 타지 않고, 전체 텍스트를 LLM에 직접 전달하여 요약합니다.
                # (기존 RAG 체인을 재사용하려다 내부 변수 불일치 오류가 발생했으므로 단순화)
                
                # LLM 직접 호출
                messages = [
                    ("system", """당신은 한국 연말정산 및 세법 전문가입니다. 
아래 제공된 [2025년 개정세법 전문]을 바탕으로 질문에 대해 상세히 요약해서 답변하세요.
내용이 많으므로 핵심적인 변화 위주로, 카테고리별로 잘 정리해서 답변하세요.

[2025년 개정세법 전문]:
""" + full_2025_text),
                    ("human", prompt)
                ]
                
                # 별도의 LLM 인스턴스 사용
                temp_llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
                response_msg = temp_llm.invoke(messages)
                response = response_msg.content
                
                # 출처 표기용 가짜 context
                sources = [Document(page_content="2025년 개정세법 전체 데이터 (요약 모드)", metadata={"source": "data/2025년 개정세법.txt"})]
                response_dict = {"answer": response, "context": sources}

            else:
                # 일반적인 RAG 실행
                response_dict = chain.invoke({"input": prompt, "chat_history": chat_history})
                response = response_dict["answer"]
                sources = response_dict.get('context', [])
            
            # 출처 표기 (Source attribution)
            if sources:
                with st.expander("참고 자료"):
                    seen_sources = set()
                    for i, doc in enumerate(sources):
                        source_name = doc.metadata.get('source', 'Unknown')
                        if source_name not in seen_sources:
                            st.write(f"**출처:** {os.path.basename(source_name)}")
                            st.caption(doc.page_content[:200] + "...")
                            seen_sources.add(source_name)

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
