import streamlit as st

st.markdown("# QA Service")
st.sidebar.markdown("# 🤖 QA Service")

st.text_input('Ask your question below')

if st.button('Submit'):
    st.text("Hmmm......I'm not sure I understand your question.")