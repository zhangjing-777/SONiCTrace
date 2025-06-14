from langchain.prompts import PromptTemplate

# define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a SONiC CLI expert. Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}

Answer in concise technical English:"""
)