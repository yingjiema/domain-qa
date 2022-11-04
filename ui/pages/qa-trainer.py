import streamlit as st
import pandas as pd
import requests
import os
import boto3
from botocore.exceptions import ClientError


BUCKET_NAME = 'jason-jones-learned-his-lesson'

SERVICE_IP = 'http://gpl:8000/'
train_endpoint = SERVICE_IP + 'train-retriever'
save_endpoint = SERVICE_IP + 'store-retriever'

SQLITE_FILE = 'sqlite_faiss_document_store.db'
SQL_URL = 'sqlite:///' + SQLITE_FILE

def file_to_list(contents):
    contents = contents.decode('utf-8').split('\n\n')
    return contents

st.set_page_config(page_title='Domain QA: Bot Trainer')

st.sidebar.markdown("# 🏋️ Bot Trainer")
main_heading = 'Domain QA: Bot Trainer'
st.markdown('## ' + main_heading )

st.markdown('### Step 1: Name your project')
proj_name = st.text_input('Project name')

st.markdown('### Step 2: Upload your data')
st.write('Upload the file you would like to train your QA Engine on!')

uploaded_file = st.file_uploader("Choose a file")

st.markdown('### Step 3: Start training your bot!')

if st.button('Submit'):
    if uploaded_file is not None:
        if proj_name is None:
            proj_name = 'untitled'
        e = None
        s3 = boto3.client('s3')
        proj_folder = 'projects/' + proj_name.lower() + '/'
        try:
            with st.spinner('Sending files to the cloud...'):
                response = s3.upload_fileobj(uploaded_file, BUCKET_NAME, proj_folder + uploaded_file.name)
                st.write(BUCKET_NAME, '/projects/' + proj_name.lower() + '/' + uploaded_file.name)
        except ClientError as e:
            st.exception(e)

        if e is None:
            st.markdown('Your data has been successfully uploaded!')

        with st.spinner('Training the retriever...'):
            train_details = {'file_name': uploaded_file.name, 'proj_name': proj_name.lower(), 'sql_url':  SQL_URL}
            response = requests.post(train_endpoint, json=train_details)
            st.write(response)


            st.write(response)

st.markdown('### Step 4: Store the trained model')
if st.button('Store file'):
    with st.spinner('Uploading trained files...'):
        save_details = {'file_name': SQLITE_FILE, 'proj_name': proj_name.lower() }
        response = requests.post(save_endpoint, json=save_details)