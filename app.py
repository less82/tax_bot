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
    # 1. Setup Vector Store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 2. Setup LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    # 3. Contextualize Question (History Aware Retriever)
    # This chain rewrites the question based on history to make it standalone
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

    # 4. Answer Question (Stuff Documents Chain)
    # This chain takes the documents and the question and generates the answer
    qa_system_prompt = """당신은 한국 세무 전문가입니다. 다음 문맥(context)을 사용하여 질문에 답변하세요.
    
    문맥:
    {context}
    
    규칙:
    1. 2025년 개정세법 내용이 있다면 이를 최우선으로 반영하세요.
    2. 문맥에 없는 내용은 지어내지 말고 "제공된 자료에서 관련 내용을 찾을 수 없습니다"라고 답하세요.
    3. 친절하고 이해하기 쉽게 설명하세요.
    4. 관련된 내용이 2024년 자료와 2025년 개정안에 모두 있다면, 개정안을 기준으로 설명하고 변경 전 내용도 간략히 언급해주세요.
    
    답변은 한국어로 작성하세요."""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 5. Final Retrieval Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

if "store" not in st.session_state:
    st.session_state.store = {}

session_id = "user_session"
if session_id not in st.session_state.store:
    st.session_state.store[session_id] = ChatMessageHistory()

# Helper to manage history manually since Streamlit reruns
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state.store[session_id]

chain = get_chain()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("연말정산에 대해 궁금한 점을 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성 중입니다..."):
            
            # Convert streamlit history to LangChain history format for current run
            current_history = get_session_history(session_id)
            # (Optional: Sync session_state messages to history object if needed, 
            # but usually RunnableWithMessageHistory handles persistence. 
            # Here we just use the manual history passing for simplicity or update it.)
            
            # Since we are managing history manually in session_state for UI, 
            # we need to ensure the chain gets the correct history format.
            # However, create_retrieval_chain expects 'chat_history' in input 
            # if we don't use RunnableWithMessageHistory wrapper.
            # Let's manually construct chat_history from session_state for the invoke.
            
            from langchain_core.messages import HumanMessage, AIMessage
            chat_history = []
            for msg in st.session_state.messages[:-1]: # Exclude current prompt
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            response_dict = chain.invoke({"input": prompt, "chat_history": chat_history})
            response = response_dict["answer"]
            
            # Source attribution
            sources = response_dict.get('context', [])
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
