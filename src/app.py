# app.py

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA

# ---- Load Components ----

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector DB
vector_db = FAISS.load_local("notebooks/vector_store", embedding_model, allow_dangerous_deserialization=True)

# Load LLM (small model for fast demo)
qa_pipeline = pipeline("text-generation", model="distilgpt2", max_length=200)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Create RAG Chain
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever(search_kwargs={"k": 5}))

# ---- Streamlit UI ----
st.set_page_config(page_title="CrediTrust Complaint Chatbot", layout="centered")
st.title(" CrediTrust Complaint Assistant")
st.markdown("Ask a question about customer complaints across Credit Cards, BNPL, etc.")

# Input box
user_question = st.text_input("Enter your question:")

# Ask button
if st.button("Ask"):
    if user_question:
        with st.spinner("Thinking..."):
            result = rag_chain.invoke({"query": user_question})
            st.success("Answer:")
            st.write(result["result"])

            # Show top retrieved source chunks
            st.markdown("---")
            st.subheader("Top 2 Retrieved Chunks (Sources)")
            top_docs = vector_db.similarity_search(user_question, k=2)
            for i, doc in enumerate(top_docs):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content)

# Clear button
if st.button("Clear"):
    st.experimental_rerun()
