import streamlit as st
import pandas as pd
import requests
import os
import boto3
from botocore.exceptions import ClientError


BUCKET_NAME = 'jason-jones-learned-his-lesson'


# pip install awswrangler
SERVICE_IP = os.getenv('SERVICE_IP')
service_endpoint = 'http://{}:8003/train-retriever'.format(SERVICE_IP)

def file_to_list(contents):
    contents = contents.decode('utf-8').split('\n\n')
    return contents

st.set_page_config(page_title='Domain QA: Bot Trainer')

st.sidebar.markdown("# 🏋️ Bot Trainer")
main_heading = 'Domain QA: Bot Trainer'
st.markdown('## ' + main_heading )

st.markdown('### Step 1: Name your project')
pname = st.text_input('Project name')

st.markdown('### Step 2: Upload your data')
st.write('Upload the file you would like to train your QA Engine on!')

uploaded_file = st.file_uploader("Choose a file")

st.markdown('### Step 3: Start training your bot!')

if st.button('Submit'):
    if uploaded_file is not None:
        if pname is None:
            pname = 'untitled'
        e = None
        s3 = boto3.client('s3')
        try:
            with st.spinner('Sending files to the cloud...'):
                response = s3.upload_fileobj(uploaded_file, BUCKET_NAME, 'projects/' + pname.lower() + '/' + uploaded_file.name)
                st.write(BUCKET_NAME, '/projects/' + pname.lower() + '/' + uploaded_file.name)
        except ClientError as e:
            st.exception(e)

        if e is None:
            st.markdown('Your data has been successfully uploaded!')

        with st.spinner('Training the retriever...'):
            response = requests.post('http://127.0.0.1:8003/train-retriever', json= {'file_name': uploaded_file.name })
            st.write(response)
