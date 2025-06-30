# 🤖 AI Document Assistant Using Llama3

Your intelligent document analysis companion powered by LLaMA 3 and advanced RAG (Retrieval-Augmented Generation) technology.

![image](https://github.com/user-attachments/assets/40aaa7d8-7ceb-446a-9d37-8f6e145eb0de)

## 📋 Overview

AI Document Assistant is a Streamlit-based web application that allows users to upload PDF documents and interact with them through natural language queries. The application uses advanced AI techniques to understand document content and provide accurate, contextual responses based on the uploaded materials.

## ✨ Features

- *📄 PDF Document Upload*: Support for multiple PDF file uploads
- *🔍 Intelligent Search*: Vector-based document retrieval using FAISS
- *💬 Conversational Interface*: Natural language Q&A with your documents
- *📚 Source Attribution*: Track which documents and pages answers come from
- *💾 Conversation History*: Persistent chat history with timestamps
- *🎨 Modern UI*: Clean, responsive interface with custom styling
- *⚙ Configurable*: Customizable system prompts and processing parameters

## 🛠 Technology Stack

- *Frontend*: Streamlit
- *LLM*: LLaMA 3 (via Ollama)
- *Vector Database*: FAISS
- *Document Processing*: PyPDF2
- *Text Chunking*: LangChain RecursiveCharacterTextSplitter
- *Embeddings*: Ollama Embeddings
- *Framework*: LangChain

## 📦 Installation

### Prerequisites

1. *Python 3.8+* installed on your system
2. *Ollama* installed and running locally
3. *LLaMA 3 model* pulled in Ollama

### Step 1: Install Ollama

Visit [Ollama's official website](https://ollama.ai/) and follow the installation instructions for your operating system.

### Step 2: Pull LLaMA 3 Model

bash
ollama pull llama3:instruct


### Step 3: Clone the Repository

bash
git clone https://github.com/yourusername/ai-document-assistant.git
cd ai-document-assistant


### Step 4: Install Python Dependencies

bash
pip install -r requirements.txt


## 📋 Dependencies

Create a requirements.txt file with the following dependencies:

txt
streamlit==1.28.0
PyPDF2==3.0.1
langchain==0.0.340
langchain-community==0.0.1
faiss-cpu==1.7.4
ollama==0.1.7


## 🚀 Usage

### Starting the Application

1. *Start Ollama server* (if not already running):
   bash
   ollama serve
   

2. *Run the Streamlit application*:
   bash
   streamlit run app.py
   

3. *Open your browser* and navigate to http://localhost:8501

### Using the Application

1. *Upload Documents*: Use the sidebar to upload one or more PDF files
2. *Process Documents*: Click "Process Documents" to create vector embeddings
3. *Ask Questions*: Type your questions in the chat interface
4. *View Results*: Get AI-powered answers with source attribution
5. *Check History*: Expand the conversation history to see previous interactions

## ⚙ Configuration

### System Prompt
Customize the AI's behavior by modifying the system prompt in the sidebar. The default prompt instructs the AI to be helpful and accurate while acknowledging uncertainty.

### Processing Parameters
- *Chunk Size*: 500 characters (optimized for better context)
- *Chunk Overlap*: 50 characters
- *Retrieved Documents*: 3 most relevant chunks per query

### Model Configuration
- *Embedding Model*: llama3:instruct
- *LLM Model*: llama3:instruct
- *Vector Store*: FAISS with local persistence

## 📁 Project Structure

ai-document-assistant/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data/
│   └── vectorstore/      # FAISS vector store data
│       └── my_store/     # Persistent embeddings and conversation history
└── output/               # Output folder for generated files
    ├── output1.png       # First PNG output file
    ├── output2.png       # Second PNG output file
    ├── output3.png       # Third PNG output file
    └── output4.mp4       # MP4 output file

## 🔧 Key Functions

- read_data(): Extract text from uploaded PDF files
- get_chunks(): Split documents into manageable chunks
- vector_store(): Create and save FAISS vector embeddings
- user_input(): Process user queries and generate responses
- get_conversational_chain(): Create RAG chain for document Q&A

## 🎯 Use Cases

- *Research Analysis*: Quickly extract insights from academic papers
- *Document Summarization*: Get key points from lengthy reports
- *Legal Document Review*: Query specific clauses or terms
- *Technical Documentation*: Find specific procedures or requirements
- *Educational Materials*: Study and quiz from textbooks or notes

## 🔒 Privacy & Security

- All processing happens locally on your machine
- No data is sent to external services (except Ollama if configured remotely)
- Documents and conversations are stored locally in the data/ directory
- Vector embeddings are persisted locally using FAISS

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- *Ollama* for providing local LLM infrastructure
- *LangChain* for the RAG framework
- *Streamlit* for the web interface
- *Meta* for the LLaMA model
- *Facebook AI Research* for FAISS vector search

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/ai-document-assistant/issues) page
2. Create a new issue with detailed information
3. Ensure Ollama is running and the LLaMA 3 model is properly installed

## 🚧 Roadmap

- [ ] Support for additional document formats (Word, Excel, PowerPoint)
- [ ] Multiple LLM model support
- [ ] Advanced chunking strategies
- [ ] Document comparison features
- [ ] Export conversation history
- [ ] Docker containerization
- [ ] Cloud deployment options

---

*Made with ❤ using Streamlit and LangChain | Powered by LLaMA 3 🤖*
