"""
Language Model (LLM) configuration and setup.

This module configures the language model for the RAG system:
- Uses OpenAI's ChatOpenAI model
- Configures model parameters and API settings
- Provides a consistent interface for text generation

The LLM is used for generating responses based on retrieved context
and user queries.
"""

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
