"""
RAG (Retrieval-Augmented Generation) Agent implementation.

This module implements a RAG agent that combines:
- Document retrieval from vector store
- Context-aware question answering
- Chat history management
- Memory persistence using Supabase

The agent maintains conversation history and provides context-aware responses
by retrieving relevant documents and using them to generate answers.
"""

from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables.base import Runnable
#from langchain_core.runnables.cache import SQLiteCache
from .retriever import retriever
from .prompting import prompt
from .llm import llm
from supabase import create_client
import datetime
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

# Supabase Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Enable LangChain
#Runnable.set_default_cache(SQLiteCache("cache/sonic_agent_cache.sqlite"))



# Memory Persistence
def persist_memory(session_id: str, messages):
    for m in messages:
        supabase.table("chat_memory_log").insert({
            "session_id": session_id,
            "message_type": m["type"],
            "message_content": m["content"],
            "timestamp": datetime.datetime.utcnow().isoformat()
        }).execute()

# RAGAgent Class
class RAGAgent:
    def __init__(self, vb_table_name: str, history_limit: int = 10):
        self.retriever = retriever(vb_table_name)
        self.parser = StrOutputParser()
        self.history_limit = history_limit

        self.chain = (
            RunnableMap({
                "context": lambda x: self.retriever.get_relevant_documents(x["query"]),
                "query": lambda x: x["query"],
                "chat_history": lambda x: self._get_chat_history(x["session_id"])
            })
            | RunnableLambda(self._merge_docs)
            | prompt
            | llm
            | self.parser
        )

    def _get_chat_history(self, session_id):
        res = supabase.table("chat_memory_log").select("*")\
            .eq("session_id", session_id).order("timestamp", desc=False).execute()

        if not res.data:
            return None

        # limit last N histories（each=human+ai）
        limited = res.data[-2 * self.history_limit:]
        history_lines = [f"{m['message_type']}: {m['message_content']}" for m in limited]
        return "\n".join(history_lines)

    def _merge_docs(self, x):
        return {
            "query": x["query"],
            "chat_history": x["chat_history"],
            "context": "\n\n".join([doc.page_content for doc in x["context"]])
        }

    def run(self, query: str, session_id: str = None) -> dict:
        session_id = session_id or str(uuid.uuid4())
        persist_memory(session_id, [{"type": "human", "content": query}])
        chat_history = self._get_chat_history(session_id)
        result = self.chain.invoke({
            "query": query,
            "session_id": session_id,
            "chat_history": chat_history
        })
        persist_memory(session_id, [{"type": "ai", "content": str(result)}])
        return result
