import streamlit as st
import pandas as pd
import requests
import os
import boto3


st.set_page_config(page_title='Domain QA')

st.sidebar.markdown("# Home")
main_heading = 'Welcome to QA Domain!'
st.markdown('## ' + main_heading )

st.text('Ask one of our bots a question, or train your own bot!')