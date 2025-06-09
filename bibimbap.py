import streamlit as st

st.set_page_config(page_title='비빕밥 레시피 상담 챗봇', page_icon='🥗')
st.title('🥗비빔밥 레시피 상담 챗봇')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

print (f'before: {st.session_state.message_list}')

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

placeholder = '궁금한 비빔밥 레시피에 대하여 질문을 작성해주세요.'
if user_question := st.chat_input(placeholder=placeholder):
    with st.chat_message('user'):
        ## 사용자 메시지 화면 출력
        st.write(user_question)
    st.session_state.message_list.append({'role':'user', 'content': user_question})

with st.chat_message('ai'):
    st.write('여기는 ai 메시지')
st.session_state.message_list.append({'role': 'ai', 'content': '여기는 AI 메시지'})


print(f'after: {st.session_state.message_list}')

print(user_question)