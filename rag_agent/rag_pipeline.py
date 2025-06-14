from langchain.chains import RetrievalQA
from .llm import llm
from .retriever import retriever
from .prompting import prompt_template


# 构造 RAG QA chain with prompt
def qa_chain(vb_table_name):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever(vb_table_name),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

