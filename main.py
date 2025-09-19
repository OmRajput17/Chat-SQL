import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()
### Setup LANGCHAIN TRACING
# Load from Streamlit secrets
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")


### Setting up streamlit app

st.set_page_config(page_title="Chal SQL", page_icon="🦜")
st.title("🦜 Chat SQL : Chat with your DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

### Radio option
radio_opt = ["Use SQLLite3 Database - student.db", "Connect to your own MySQL database"]

selected_opt = st.sidebar.radio(label="Choose the DB", options=radio_opt)


if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide my SQL Host Name")
    mysql_user = st.sidebar.text_input("MYSQL User")
    mysql_password = st.sidebar.text_input("MYSQL Password", type="password")
    mysql_db = st.sidebar.text_input("MYSQL DATABASE")
else:
    db_uri = LOCALDB

if not db_uri:
    st.info("Please enter the DB info and uri")

if not groq_api_key:
    st.info("Please enter a valid API key")

llm = ChatGroq(groq_api_key = groq_api_key, model_name="llama-3.1-8b-instant", streaming = True)


#### Connectiong to the DATABASE
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host = None,mysql_user = None , mysql_password = None , mysql_db = None):
    if db_uri == LOCALDB:
        dbFile_path = (Path(__file__).parent/"student.db").absolute()
        print(dbFile_path)
        creator = lambda : sqlite3.connect(f"file:{dbFile_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator = creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.info("Please provide all MYSQL Connection Details")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

if db_uri==MYSQL:
    db = configure_db(
        db_uri=db_uri,
        mysql_host= mysql_host,
        mysql_user=mysql_user,
        mysql_password=mysql_password,
        mysql_db=mysql_db,
    )
else:
    db = configure_db(db_uri=db_uri)


### Agent and Tool Kit

toolkit = SQLDatabaseToolkit(
    db=db, 
    llm=llm,
) 

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

### Creating Session State

if "messages" not in st.session_state or st.sidebar.button("Clear Message History"):
    st.session_state["messages"] = [{"role":"assistant", "content":"How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

### User_Query
user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role":"user", "content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)
