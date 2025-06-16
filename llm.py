import os
import json

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (ChatPromptTemplate, FewShotPromptTemplate,
                                    MessagesPlaceholder, PromptTemplate)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import answer_examples



##환경변수 읽어오기 ======================================
load_dotenv()

## llm 생성 ==============================================
def load_llm(model='gpt-4o'):
    return ChatOpenAI(model=model)

## Embedding 설정 + Vector Store Index 가져오기 =========================
def load_vectorstore(index_name = 'bibimbap'):
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

    ## 임베딩 모델 지정
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    Pinecone(api_key=PINECONE_API_KEY)

    #저장된 인덱스 가져오기
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )


### 세션별 히스토리 저장 ===============================
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


## 히스토리 기반 리트리버 ==============================
def build_history_aware_retriever(llm, retriever):
    contextualize_q_system_prompt = (
        "채팅 기록과 사용자의 최근 질문을 바탕으로, 과거 대화를 반영한 독립적인 질문을 생성하세요."
        "질문에 답변하지 말고, 필요한 경우 질문을 재구성하세요."
        "이해할 수 있는 독립적인 질문을 작성하십시오."
        "채팅 기록 없이도 가능합니다. 질문에 답변하지 마십시오."
        "필요한 경우 질문을 재구성하고, 그렇지 않은 경우 그대로 반환하십시오."
    )

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

    return history_aware_retriever

def build_few_shot_examples() -> str:
    example_prompt = PromptTemplate.from_template("Question: {input}\n\nAnswer: {answer}") # 단일

    few_shot_prompt = FewShotPromptTemplate(
        examples=answer_examples,           ## 질문/답변 예시들 (전체 type은 list, 각 질문/답변 type은 dict)
        example_prompt=example_prompt,      ## 단일 예시 포맷
        prefix='다음 질문에 답변하세요 : ', ## 예시들 위로 추가되는 텍스트(도입부)
        suffix="Question: {input}",         ## 예시들 뒤로 추가되는 텍스트(실제 사용자 질문 변수)
        input_variables=["input"],          ## suffix에서 사용할 변수
    )

    formmated_few_shot_prompt = few_shot_prompt.format(input='{input}')

    return formmated_few_shot_prompt

## [외부 사전 로드] ============================================================

def load_dictionary_from_file(path = 'keyword-dictionary.json'):
    with open(path, 'r', encoding='utf-8')as file:
        return json.load(file)

def build_dictionary_text(dictionary: dict) -> str:
    return '\n'.join([
        f'{k} ({",".join(v["tags"])}): {v["definition"]} [출처: {v["source"]}]'
        for k, v in dictionary.items()
    ])

## QA prompt ====================================================================
def build_qa_prompt() :
    keyword_dictionary = load_dictionary_from_file()
    dictionary_text = build_dictionary_text(keyword_dictionary)


    system_prompt = (
        '''[identity]

-당신은 비빔밥 레시피 전문가 입니다.
-[context]를 참고하여 사용자의 질문에 답변하세요.
-답변은 출처에 해당하는 재료와 레시피 순서 형식으로 답변해주세요.
-항목별로 표시해서 답변해주세요.
-비빔밥 레시피 이외에는 '비빔밥 레시피와 관련된 질문을 해주세요.'로 답하세요.
-필요한 경우 이모티콘을 활용해서 귀엽게 해주세요.
[context] 
{context} 

[keyword_dictionary]
{dictionary_text}
'''
    )
    
    formmated_few_shot_prompt = build_few_shot_examples()
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", system_prompt),
        ('assistant', formmated_few_shot_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ]
    ).partial(dictionary_text=dictionary_text)

    print(f'\nqa_prompt >>\n{qa_prompt.partial_variables}')

    return qa_prompt

## 전체 chain 구성 =================================================
def build_conversational_chain():
    LANCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')


    ## LLM 모델 지정
    llm  = load_llm()

    ##vector store에서 index 정보
    database = load_vectorstore()
    retriever = database.as_retriever(search_kwargs={'k': 3})

    history_aware_retriever = build_history_aware_retriever(llm, retriever)
    
    qa_prompt = build_qa_prompt()
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    return conversational_rag_chain


## [AI Message 함수 정의] ######################
def stream_ai_message(user_message, session_id='default'):
    qa_chain = build_conversational_chain()

    ai_message = qa_chain.stream(
        {"input" : user_message},
        config={"configurable": {"session_id": session_id}},
    )

    print(f'대화이력 -> , {get_session_history(session_id)}\n\n')
    print('=' * 50 + '\n')
    print(f'[stream_ai_message 함수 내 출력] session_id>> {session_id}')
    
    ########################################################################
    ## vector store에서 검색된 문서 출력
    retriever = load_vectorstore().as_retriever(search_kwargs={'k': 1})
    search_results = retriever.invoke(user_message)
    if search_results:
        print(f'\nPinecone 검색 결과 >> \n{search_results[0].page_content[:100]}')

    else:
        print('\n❌ 검색 결과가 없습니다. 사전과 예시를 기반으로 GPT가 답변합니다.')
    ########################################################################

    return ai_message

