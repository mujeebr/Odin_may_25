import streamlit as st
import os
import nltk
import redis

from langchain.cache import RedisCache
from langchain.globals import set_llm_cache
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# --- Setup Redis cache for LLM (optional, not used with manual caching) ---
redis_client = redis.Redis(host="localhost", port=6379, db=0)
# set_llm_cache(RedisCache(redis_client))  # not needed if using manual cache

# --- Download NLTK tokenizer ---
nltk.download('punkt')

# --- Set OpenAI API key ---
OPENAI_API_KEY = st.password_input("Enter your OpenAI API key:", type="password")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Prompt template ---
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# --- Load Excel ---
def load_excel(file_path):
    loader = UnstructuredExcelLoader(file_path)
    return loader.load()

# --- Split documents ---
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

# --- Build semantic retriever ---
def build_semantic_retriever(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)
    return vector_store.as_retriever()

# --- Build BM25 retriever ---
def build_bm25_retriever(documents):
    return BM25Retriever.from_documents(documents, preprocess_func=word_tokenize)

# --- Build hybrid retriever ---
def build_hybrid_retriever(documents):
    semantic_retriever = build_semantic_retriever(documents)
    bm25_retriever = build_bm25_retriever(documents)
    return EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.3, 0.7]
    )

# --- Get answer from cache or LLM ---
def get_or_cache_answer(question, documents):
    cache_key = f"question:{question.strip().lower()}"
    cached_answer = redis_client.get(cache_key)

    if cached_answer:
        return cached_answer.decode("utf-8")

    # If not cached, generate answer
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | ChatOpenAI(model="gpt-4")
    response = chain.invoke({"question": question, "context": context})

    # Cache the result
    redis_client.set(cache_key, response.content)
    return response.content

# --- Streamlit UI ---
st.title("Excel QA App with Hybrid Retriever and Redis Cache")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

# Load and cache documents and retriever
if uploaded_file:
    file_path = os.path.join("./", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "retriever" not in st.session_state:
        documents = load_excel(file_path)
        chunked_documents = split_text(documents)
        retriever = build_hybrid_retriever(chunked_documents)

        st.session_state.documents = chunked_documents
        st.session_state.retriever = retriever

# Chat input for Q&A
question = st.chat_input("Ask a question about the uploaded Excel file...")

if question and "retriever" in st.session_state:
    st.chat_message("user").write(question)

    related_docs = st.session_state.retriever.invoke(question)
    answer = get_or_cache_answer(question, related_docs)

    st.chat_message("assistant").write(answer)
