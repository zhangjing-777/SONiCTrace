"""
RAG (Retrieval-Augmented Generation) Pipeline implementation.

This module provides the core RAG pipeline that:
- Combines retrieval and generation components
- Uses a custom prompt template
- Integrates with the vector store retriever
- Connects with the language model

The pipeline is designed to provide accurate and context-aware responses
by retrieving relevant documents before generating answers.
"""

from langchain.chains import RetrievalQA
from .llm import llm
from .retriever import retriever
from .prompting import prompt_template


# build RAG QA chain with prompt
def qa_chain(vb_table_name):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever(vb_table_name),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

