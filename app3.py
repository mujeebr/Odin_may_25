# import os
# import streamlit as st
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate

# # Set up directory to store PDFs
# pdf_dir = "pdfs"
# os.makedirs(pdf_dir, exist_ok=True)

# # Set the app title
# st.title("PDF RAG Chatbot")

# # Sidebar for API key input
# st.sidebar.header("Configuration")
# api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password")

# # File uploader for PDFs
# st.header("Upload PDF(s) and Ask Questions")
# uploaded_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=["pdf"])

# # Save uploaded PDFs into the "pdfs" folder
# if uploaded_files:
#     for file in uploaded_files:
#         file_path = os.path.join(pdf_dir, file.name)
#         with open(file_path, "wb") as f:
#             f.write(file.getbuffer())
#     st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")

# # User question input
# question = st.text_input("Ask a question related to the PDFs:")

# if api_key and uploaded_files:
#     try:
#         # Load PDFs from the directory
#         loader = PyPDFDirectoryLoader(pdf_dir)
#         data = loader.load()

#         # Split text into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
#         chunks = text_splitter.split_documents(data)

#         # Create embeddings and vector store
#         embedding_model = OpenAIEmbeddings(api_key=api_key)
#         vectordatabase = FAISS.from_documents(chunks, embedding_model)

#         # Initialize the LLM
#         llm = OpenAI(api_key=api_key)

#         # Define prompt template
#         template = """Use the context to provide a concise answer. If you don't know, just say 'I don't know'.
#         {context}
#         Question: {question}
#         Helpful Answer:"""

#         prompt = PromptTemplate.from_template(template)

#         # Create retrieval chain
#         chain = RetrievalQA.from_chain_type(llm=llm,
#                                             chain_type="stuff",
#                                             retriever=vectordatabase.as_retriever(),
#                                             chain_type_kwargs={"prompt": prompt})

#         if question:
#             with st.spinner("Searching..."):
#                 response = chain.run(question)
#             st.subheader("Answer:")
#             st.write(response)
#     except Exception as e:
#         st.error(f"Error: {str(e)}")
# else:
#     st.warning("Please enter your OpenAI API key and upload PDFs to proceed.")

# import os
# import streamlit as st
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory

# # Set up directory to store PDFs
# pdf_dir = "pdfs"
# os.makedirs(pdf_dir, exist_ok=True)

# # Streamlit app title and sidebar
# st.title("PDF RAG Chatbot with Memory")
# st.sidebar.header("Configuration")
# api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password")

# # File uploader
# st.header("Upload PDF(s)")
# uploaded_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=["pdf"])

# if uploaded_files:
#     for file in uploaded_files:
#         file_path = os.path.join(pdf_dir, file.name)
#         with open(file_path, "wb") as f:
#             f.write(file.getbuffer())
#     st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")

# # Memory for chat history
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # Chat input
# with st.form(key="chat_form"):
#     st.header("Ask a Question")
#     question = st.text_input("Enter your question:")
#     submit_button = st.form_submit_button(label="Submit")

# if submit_button and api_key and uploaded_files:
#     try:
#         # Load and split PDF documents
#         loader = PyPDFDirectoryLoader(pdf_dir)
#         documents = loader.load()

#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         chunks = splitter.split_documents(documents)

#         # Embeddings + FAISS vectorstore
#         embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#         vectordb = FAISS.from_documents(chunks, embeddings)

#         # OpenAI LLM
#         llm = OpenAI(openai_api_key=api_key)

#         # Build Conversational Retrieval Chain with memory
#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             retriever=vectordb.as_retriever(),
#             memory=memory,
#             return_source_documents=False
#         )

#         with st.spinner("Thinking..."):
#             response = qa_chain.run(question)

#         st.subheader("Answer:")
#         st.write(response)

#     except Exception as e:
#         st.error(f"Error: {str(e)}")
# elif submit_button:
#     st.warning("Please enter your OpenAI API key and upload PDFs to proceed.")


import os
import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set up directory to store PDFs
pdf_dir = "abf"
os.makedirs(pdf_dir, exist_ok=True)

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 PDF RAG Chatbot with Memory")

# Sidebar for API key
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Upload PDFs
st.header("Upload PDF(s)")
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(pdf_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")

# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Build Vector Store and Chain once
if api_key and uploaded_files and st.session_state.qa_chain is None:
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = FAISS.from_documents(chunks, embeddings)

    llm = OpenAI(openai_api_key=api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=False
    )

# Display full-width chat interface
st.subheader("💬 Chat with your PDFs")
user_input = st.text_input("Ask a question", key="input")

if st.button("Send") and user_input:
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        response = st.session_state.qa_chain.run(user_input)
        st.session_state.chat_history.append((user_input, response))

# Display chat history
for question, answer in st.session_state.chat_history:
    st.markdown(f"**You:** {question}")
    st.markdown(f"**Bot:** {answer}")
    st.markdown("---")
