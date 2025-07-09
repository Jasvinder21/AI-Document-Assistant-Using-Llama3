import streamlit as st
from PyPDF2 import PdfReader
import os
import tempfile
import json
from datetime import datetime
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM as Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import ollama

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(files):
    documents = []
    progress_bar = st.progress(0)
    total_files = len(files)
    
    for idx, file in enumerate(files):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        try:
            pdf_reader = PdfReader(tmp_file_path)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                documents.append(Document(page_content=text, metadata={"source": file.name, "page_number": page_num + 1}))
        finally:
            os.remove(tmp_file_path)
        
        # Update progress
        progress = (idx + 1) / total_files
        progress_bar.progress(progress)
    
    return documents

def get_chunks(texts, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for text in texts:
        split_texts = text_splitter.split_text(text.page_content)
        for split_text in split_texts:
            chunks.append(Document(page_content=split_text, metadata=text.metadata))
    return chunks

def vector_store(text_chunks, embedding_model_name, vector_store_path):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.from_texts(
            texts=[doc.page_content for doc in text_chunks],
            embedding=embeddings,
            metadatas=[doc.metadata for doc in text_chunks]
        )
        vector_store.save_local(vector_store_path)
        logger.info(f"Vector store created and saved at {vector_store_path}")
    except Exception as e:
        logger.error(f"Failed to create vector store with {embedding_model_name}: {str(e)}")
        st.error(f"Failed to initialize embeddings: {str(e)}. Please try a different model or check your network.")
        raise

def load_vector_store(embedding_model_name, vector_store_path):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"Vector store loaded from {vector_store_path}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load vector store with {embedding_model_name}: {str(e)}")
        st.error(f"Failed to load vector store: {str(e)}. Please process documents again.")
        raise

def save_conversation(conversation, vector_store_path):
    conversation_path = os.path.join(vector_store_path, "conversation_history.json")
    with open(conversation_path, "w") as f:
        json.dump(conversation, f, indent=4)

def load_conversation(vector_store_path):
    conversation_path = os.path.join(vector_store_path, "conversation_history.json")
    if os.path.exists(conversation_path):
        with open(conversation_path, "r") as f:
            conversation = json.load(f)
    else:
        conversation = []
    return conversation

def document_to_dict(doc):
    return {
        "metadata": doc.metadata
    }

def check_model_availability(model_name):
    try:
        models = ollama.list()['models']
        return any(model['name'].startswith(model_name) for model in models)
    except Exception as e:
        logger.error(f"Failed to check model availability: {str(e)}")
        return False

def get_conversational_chain(retriever, ques, llm_model, system_prompt):
    if not check_model_availability(llm_model):
        logger.warning(f"Model {llm_model} not found on Ollama server.")
        return {"result": f"Sorry, the model '{llm_model}' is not available. Please ensure the model is pulled using 'ollama pull {llm_model}' or try a different model."}
    
    try:
        llm = Ollama(model=llm_model, base_url="http://localhost:11434", verbose=True)
    except Exception as e:
        logger.warning(f"Failed to connect to Ollama LLM: {str(e)}. Using fallback response.")
        return {"result": "Sorry, the AI model is unavailable. Please check if the Ollama server is running or try again later."}
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    try:
        response = qa_chain.invoke({"query": ques})
        return response
    except Exception as e:
        logger.error(f"Failed to invoke QA chain: {str(e)}")
        return {"result": f"Error processing query: {str(e)}. Please try again or check the model configuration."}

def user_input(user_question, embedding_model_name, vector_store_path, num_docs, llm_model, system_prompt):
    try:
        vector_store = load_vector_store(embedding_model_name, vector_store_path)
        retriever = vector_store.as_retriever(search_kwargs={"k": num_docs})
        response = get_conversational_chain(retriever, user_question, llm_model, system_prompt)
        
        conversation = load_conversation(vector_store_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if 'result' in response:
            result = response['result']
            source_documents = response['source_documents'] if 'source_documents' in response else []
            conversation.append({
                "question": user_question,
                "answer": result,
                "timestamp": timestamp,
                "llm_model": llm_model,
                "source_documents": [document_to_dict(doc) for doc in source_documents]
            })
            
            st.success("AI Assistant's Response:")
            st.write(result)
            st.info(f"🤖 Model: {llm_model}")
            
            with st.expander("📚 Source Documents", expanded=False):
                for doc in source_documents:
                    metadata = doc.metadata
                    st.write(f"📄 **Source:** {metadata.get('source', 'Unknown')}")
                    st.write(f"📃 **Page:** {metadata.get('page_number', 'N/A')}")
                    st.write(f"ℹ️ **Additional Info:** {metadata}")
                    st.markdown("---")
        else:
            conversation.append({"question": user_question, "answer": response, "timestamp": timestamp, "llm_model": llm_model})
            st.error("An error occurred. Please try again.")
        
        save_conversation(conversation, vector_store_path)
        
        with st.expander("💬 Conversation History", expanded=False):
            for entry in sorted(conversation, key=lambda x: x['timestamp'], reverse=True):
                st.write(f"🙋 **Question ({entry['timestamp']}):** {entry['question']}")
                st.write(f"🤖 **Answer:** {entry['answer']}")
                st.write(f"🧠 **Model:** {entry['llm_model']}")
                if 'source_documents' in entry:
                    for doc in entry['source_documents']:
                        st.write(f"📚 **Source:** {doc['metadata'].get('source', 'Unknown')}, **Page:** {doc['metadata'].get('page_number', 'N/A')}")
                st.markdown("---")
    except Exception as e:
        st.error(f"Error processing your request: {str(e)}. Please try again or select a different embedding model.")

def main():
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 0rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #6A798F;
            color: white;
            padding: 0.75rem;
            border-radius: 5px;
            border: none;
            font-weight: bold;
            margin-top: 1rem;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 0.5rem;
            border: 1px solid #e0e0e0;
        }
        .stTextArea>div>div>textarea {
            border-radius: 5px;
            padding: 0.5rem;
            border: 1px solid #e0e0e0;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
            padding: 20px;
        }
        .sidebar .sidebar-content .stMarkdown h3 {
            margin-top: 0;
        }
        h1 {
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            padding-top: 1rem;
        }
        h2 {
            color: #555;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stExpander {
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
        }
        .header-container {
            background-color: #D3D3D3;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .header-container h1,
        .header-container h2 {
            color: #333;
            margin: 0;
            padding: 0;
        }
        .sidebar-header {
            background-color: #D3D3D3;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 1rem;
            text-align: center;
        }
        .sidebar-header h3 {
            color: #333;
            margin: 0;
            padding: 0;
        }
        .sidebar-header .llama-emoji {
            font-size: 1.5rem;
            line-height: 1;
            display: inline-block;
            vertical-align: middle;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main content
    st.markdown(
        """
        <div class="header-container">
            <h1>🤖 AI Document Assistant</h1>
            <h2>Your intelligent document analysis companion</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Configuration variables
    embedding_models = ["all-MiniLM-L6-v2", "all-distilroberta-v1"]
    llm_model = "llama3"  # Changed from llama3:instruct to llama3
    vector_store_path = os.path.join(os.getcwd(), "data", "vectorstore", "my_store")
    os.makedirs(vector_store_path, exist_ok=True)

    chunk_text = True
    chunk_size = 500
    chunk_overlap = 50
    num_docs = 3

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-header">
                <h3><span class="llama-emoji">🦙</span> AI Doc</h3>
                <p>Intelligent Document Analysis</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("### ⚙️ Configuration")
        embedding_model_name = st.selectbox(
            "Embedding Model",
            embedding_models,
            index=0,
            help="Select the embedding model for document processing. If one fails, try another."
        )
        system_prompt = st.text_area(
            "System Prompt",
            value="You are an intelligent AI assistant designed to help users understand and analyze documents. Provide clear, concise, and accurate information based on the context given. If you're unsure about something, say so and suggest ways to find more information.",
            height=150
        )
        
        st.markdown("### 📁 Document Upload")
        data_files = st.file_uploader(
            "Upload your PDFs here",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload one or more PDF files to analyze"
        )
        
        if st.button("Process Documents", type="primary"):
            with st.spinner("🔄 Processing documents..."):
                if not data_files:
                    st.warning("Please upload at least one PDF file.")
                else:
                    raw_documents = read_data(data_files)
                    if chunk_text:
                        text_chunks = get_chunks(raw_documents, chunk_size, chunk_overlap)
                    else:
                        text_chunks = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in raw_documents]
                    try:
                        vector_store(text_chunks, embedding_model_name, vector_store_path)
                        st.success("✅ Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Failed to process documents: {str(e)}. Try a different embedding model or check your network.")

    # Main chat interface
    st.markdown("### 💬 Chat Interface")
    
    user_question = st.text_input(
        "Ask a question about your documents:",
        placeholder="E.g., What are the main topics discussed in the uploaded PDFs?",
        help="Type your question here and press Enter to get a response"
    )

    if user_question:
        with st.spinner("🧠 Thinking..."):
            user_input(user_question, embedding_model_name, vector_store_path, num_docs, llm_model, system_prompt)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>👨‍💻 Developed with ❤️ using Streamlit and LangChain</p>
            <p>Powered by Llama3 🤖</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()