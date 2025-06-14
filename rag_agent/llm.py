import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from ..config import LLM_NAME



load_dotenv()

llm = ChatOpenAI(
    model=LLM_NAME,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)
