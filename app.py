# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:33:21 2024

@author: chaki
"""

# -*- coding: utf-8 -*-

import streamlit as st


import pandas as pd

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai.chat_models import ChatOpenAI
import os


# Read the data
script_dir = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(script_dir, "../data/realestate_clean.csv")
df = pd.read_csv(csv_path)

# create a DB with sqlalchemy
engine = create_engine("sqlite:///realestate_llm.db")
df.to_sql("realestate_llm", engine, index=False, if_exists="replace") 
db = SQLDatabase(engine=engine)
print(db.dialect)
print(db.get_usable_table_names())
# Langchain Config (sqlalchemy)


st.title("🏠🔗 Use Case: LLm with SQL Database ")

st.write(""" 
         chat over a real world real estate data. Powered by OpenAI's gpt-4o & Langchain
         """)
         
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


def generate_response(input_text):
    llm = ChatOpenAI(temperature=0.7, api_key=openai_api_key, model = 'gpt-4o')
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    # model_resp = llm.invoke(input_text)
    model_resp = agent_executor.invoke({"input": input_text})
    # the returned object is a Lagnchain AIMessage similar to a JSON object, and contains a bunch of info
    # To extract the actual message:
    ai_msg = model_resp['output']
    st.info(ai_msg)





with st.form("my_form"):
    text = st.text_area(
        "Message the System:",
        f"...",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)

