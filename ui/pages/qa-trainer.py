import streamlit as st
import pandas as pd
import requests
import os
import boto3
from botocore.exceptions import ClientError

# Endpoints and S3 locations
BUCKET_NAME = 'domain-qa-system'

SERVICE_IP = 'http://gpl:8000/'

ingest_endpoint = SERVICE_IP + 'ingest'
embed_endpoint = SERVICE_IP + 'embed'
adapt_endpoint = SERVICE_IP + 'adapt'

# Helper function
def report_response(response, text):
    if response.status_code == 200:
        st.markdown(text)
    else:
        st.write(response)

#
#  Start of UI
#
st.set_page_config(page_title='Domain QA: Bot Trainer')

st.sidebar.markdown("# 🏋️ Bot Trainer")
main_heading = '🏋️ Domain QA: Bot Trainer'
st.markdown('## ' + main_heading )

#
#  Inputs and controls
#

# Initialize to default
proj_name = 'Untitled'
s3_key = 'None'

# 1. Name project
st.markdown('### Step 1: Name your project')
proj_name = st.text_input('Project name')

# 2. Data upload control
st.markdown('### Step 2: Upload your data')
st.write('Upload the file you would like to train your QA Engine on!')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    s3_key = 'projects/' + proj_name.lower() + '/' + uploaded_file.name

if st.button('Submit'):
    if uploaded_file is not None:
        
        s3_key = 'projects/' + proj_name.lower() + '/' + uploaded_file.name
        s3 = boto3.client('s3')
        
        # Upload to S3 bucket
        with st.spinner('Sending files to the cloud...'):
            response = s3.upload_fileobj(uploaded_file, BUCKET_NAME, s3_key)
            st.write(BUCKET_NAME, s3_key)

            st.markdown('✅ Your data has been successfully uploaded!')

# 3. Add data to document store
st.markdown('### Step 3: Add data to document store!')
if st.button('Store documents', key='document_store'):
    # Add to document store
    st.write(s3_key)
    with st.spinner('Adding documents to the document store...'):
        response = requests.post(ingest_endpoint, params={ "bucket": BUCKET_NAME, "key": s3_key, "index": proj_name.lower() })

    report_response(response, '✅ Documents added to document store!')

# 4. Create embeddings
st.markdown('### Step 4: Create embeddings from data!')
if st.button('Create embeddings', key='create_embeddings'):
    # Create embeddings
    with st.spinner('Creating document embeddings...'):
        response = requests.post(embed_endpoint, data={ "index": proj_name.lower() })

    report_response(response, '✅ Document embeddings created!')

# 5. Train retriever
st.markdown('### Step 5: Train retriever!')
if st.button('Train retriever', key='train_retriever'):
    # Train the retriever
    with st.spinner('Training the retriever...'):
        response = requests.post(adapt_endpoint, data={ "index": proj_name.lower() })

    report_response(response, '✅ Training is complete!')
