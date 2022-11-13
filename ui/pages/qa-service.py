# from ui.pages.qa-trainer import SERVICE_IP
import streamlit as st
import requests

ANSWER_IP = "http://demo:8000/"
answer_endpoint = ANSWER_IP + 'question-answer'

st.markdown("# 🤖 QA Service")
st.sidebar.markdown("# 🤖 QA Service")

question = st.text_input('Ask your question below')

if st.button('Submit'):
    # st.text(question)
    # data = { 'text': question }
    # response = requests.post(answer_endpoint, json=data)
    # st.text(response.status_code)
    # if response.status_code == 200:
    #     st.json(response.json())

    response = requests.post("http://ray:8000/", json=question)
    answer = response.text
    st.json(answer)

    st.text("Hmmm......I'm not sure I understand your question.")
