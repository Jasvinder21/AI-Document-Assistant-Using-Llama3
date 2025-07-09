# 🤖 AI Document Assistant Using Llama3
Your intelligent document analysis companion powered by LLaMA 3 and advanced RAG (Retrieval-Augmented Generation) technology.

![image](https://github.com/user-attachments/assets/2e533af5-cc31-4393-a9fc-0ae215c924db)
![image](https://github.com/user-attachments/assets/e70fa1a9-2b9f-43d9-a348-1a6eec5fc319)

## 📖 Overview
- The AI Document Assistant is a Streamlit-based web application that allows users to upload PDF documents and interact with them via natural language queries. Powered by LLaMA 3 and advanced RAG techniques, it provides accurate, context-aware responses, making it perfect for researchers, professionals, and students.
  
## ✨ Features

- **📄 Multi-PDF Upload**: Process multiple PDF files simultaneously.
- **🔍 Smart Search**: Vector-based retrieval using FAISS for precise results.
- **💬 Conversational Q&A**: Ask questions in natural language and get detailed answers.
- **📚 Source Attribution**: View document sources and page numbers for transparency.
- **💾 Conversation History**: Save and review chat history with timestamps.
- **🎨 Sleek UI**: Responsive, modern interface with custom styling.
- **⚙ Customizable**: Adjust embedding models, system prompts, and processing settings.

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.38.0
- **LLM**: LLaMA 3 (via Ollama)
- **Vector Database**: FAISS
- **Document Processing**: PyPDF2 3.0.1
- **Text Chunking**: LangChain RecursiveCharacterTextSplitter
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2 or all-distilroberta-v1)
- **Framework**: LangChain 0.3.1, LangChain-Community 0.3.1, LangChain-Ollama 0.2.0

## 📦 Installation
### Prerequisites

- 🐍 Python 3.8+ installed
- 🦙 Ollama installed and running locally
- 🤖 LLaMA 3 model pulled in Ollama

### Setup

1. **Install Ollama**
   Download and install Ollama from the official website for your operating system.
2. **Pull LLaMA 3 Model**
   ollama pull llama3

3. **Clone the Repository**
   ```bash
   git clone https://github.com/Jasvinder21/AI-Document-Assistant-Using-Llama3
   cd AI-Document-Assistant-Using-Llama3
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
**Notes**:Install FAISS separately (not included in requirements.txt):
   ```bach
   pip install faiss-cpu==1.8.0
   ```

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
streamlit==1.38.0
PyPDF2==3.0.1
langchain==0.3.1
langchain-community==0.3.1
langchain-ollama==0.2.0
ollama==0.3.3
sentence-transformers==3.1.1
numpy==1.26.4
faiss-cpu==1.8.0
```

## 🚀 Usage
1. **Starting the Application:**
   ```bash
   Start Ollama Server:
   ollama serve
   ```

2. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

3. **Access the App:**
   Open your browser and navigate to http://localhost:8501.
   

## Using the Application

- **📤 Upload PDFs**: Use the sidebar to upload one or more PDF files.
- **🔄 Process Documents**: Click "Process Documents" to generate vector embeddings.
- **❓ Ask Questions**: Enter queries in the chat interface.
- **📜 View Responses**: Get AI-generated answers with source document and page references.
- **🕰️ Review History**: Expand the conversation history to see past interactions.

## ⚙️ Configuration
### System Prompt
  Customize the AI's behavior via the sidebar's system prompt. The default prompt ensures           clear, accurate responses while acknowledging uncertainties.
   

### Processing Parameters

- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Retrieved Documents**: 3 relevant chunks per query

  ### Model Configuration

- **Embedding Models**: all-MiniLM-L6-v2 or all-distilroberta-v1
- **LLM Model**: LLaMA 3
- **Vector Store**: FAISS, stored locally in data/vectorstore/my_store

## 📁 Project Structure
```
AI-Document-Assistant-Using-Llama3/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data/
│   └── vectorstore/
│       └── my_store/     # FAISS vector store and conversation history
└── output/
    ├── output1.png       # Sample screenshot
    ├── output2.png       # Sample screenshot
    ├── output3.png       # Sample screenshot
    └── output4.mp4       # Demo video
```

## 🔧 Key Functions

- **read_data()**: Extracts text from PDFs using PyPDF2.
- **get_chunks()**: Splits documents into chunks for processing.
- **vector_store()**: Creates and saves FAISS vector embeddings.
- **load_vector_store()**: Loads the vector store for querying.
- **user_input()**: Handles user queries and displays responses with sources.
- **get_conversational_chain()**: Builds the RAG chain for Q&A.

## 🎯 Use Cases

- **🔬 Research Analysis**: Extract insights from academic papers or reports.
- **📝 Document Summarization**: Summarize lengthy documents quickly.
- **⚖️ Legal Review**: Query specific clauses or terms in contracts.
- **📚 Technical Docs**: Find procedures or requirements in manuals.
- **🎓 Educational Tools**: Study and quiz from textbooks or notes.

## 🔒 Privacy & Security

- **🖥️ Local Processing**: All operations run locally.
- **🔐 No External Sharing**: Documents and queries stay on your device (unless Ollama is configured remotely).
- **💾 Local Storage**: Vector embeddings and history are stored in data/.
- **🛡️ Secure Embeddings**: FAISS ensures efficient and secure vector storage.

## 🤝 Contributing
 Contributions are welcome! To contribute:

1. 🍴 Fork the repository.
2. 🌿 Create a feature branch (git checkout -b feature/amazing-feature).
3. 💾 Commit your changes (git commit -m 'Add amazing feature').
4. 🚀 Push to the branch (git push origin feature/amazing-feature).
5. 📬 Open a Pull Request.

## 📝 License

This project is licensed under the MIT License.
🙏 Acknowledgments

Ollama for local LLM infrastructure.
LangChain for RAG and text processing.
Streamlit for the web interface.
HuggingFace for embedding models.
Facebook AI Research for FAISS.

## 📞 Support
For issues or questions:

1. Visit the Issues page.
2. Create a new issue with detailed information.
3. Ensure Ollama is running and LLaMA 3 is installed (ollama pull llama3).

## 🚧 Roadmap

 Support for additional formats (DOCX, TXT).
 Multi-LLM model support.
 Advanced chunking and retrieval strategies.

 Document comparison and summarization tools.
 Exportable conversation history.
 Docker containerization.
 Cloud deployment options.


**Built with 🖥️ Streamlit & LangChain | Powered by 🦙 LLaMA 3**
