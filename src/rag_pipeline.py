# File: src/rag_pipeline.py

import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

#  Load the same embedding model used during indexing 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#  Load vector store from disk 
vector_db = FAISS.load_local("vector_store/index", embedding_model, allow_dangerous_deserialization=True)

#  Retrieve top-k similar chunks for a user query 
def retrieve_top_chunks(query: str, k: int = 5):
    """Returns top-k relevant chunks for a user question."""
    return vector_db.similarity_search(query, k=k)

#  Prompt template guiding the LLM 
rag_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use only the following complaint excerpts to formulate your answer.
If the context does not contain an answer, say you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""
)

#  LLM pipeline: Replace with more powerful models if resources allow 
llm_pipeline = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

#  RAG chain combines retriever, prompt and generator 
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": rag_prompt_template}
)

#  Run the RAG pipeline on a question 
def answer_question(question: str) -> str:
    """Returns LLM-generated answer for a user question."""
    response = rag_chain({"query": question})
    return response["result"]

#  Batch Evaluation 
def evaluate_questions(questions: list) -> pd.DataFrame:
    """Evaluates RAG on a list of questions. Returns DataFrame."""
    records = []
    for question in questions:
        result = answer_question(question)
        retrieved_docs = retrieve_top_chunks(question)
        records.append({
            "Question": question,
            "Generated Answer": result,
            "Retrieved Source 1": retrieved_docs[0].page_content[:250] if retrieved_docs else "",
            "Retrieved Source 2": retrieved_docs[1].page_content[:250] if len(retrieved_docs) > 1 else "",
            "Quality Score (1-5)": "",
            "Comments": ""
        })
    return pd.DataFrame(records)
