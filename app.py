
import streamlit as st
import pandas as pd
import os

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv

# Load API key - will pick up from .env or environment variables (e.g., Colab Secrets)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

# Streamlit UI
st.set_page_config(page_title="AI CSV Assistant")
st.title("📊 AI CSV Data Assistant")

# Upload CSV
uploaded_file = st.file_uploader("/content/Employee.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Create agent
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # User query
    query = st.text_input("Ask questions about your data:")

    if query:
    with st.spinner("Analyzing..."):
        response = agent.run(query)
        st.success(response)

# Now, run the Streamlit app. This cell will save the content above to 'app.py'
# Then, the following shell commands will run the Streamlit app and create a public URL.
# Remember to set your GROQ_API_KEY in Colab Secrets before running this.
