# RAG-Chat with Gemini 2

This project demonstrates a **Retrieval-Augmented Generation (RAG)** chat application using:
- **Google Generative AI** (Gemini model via `google.generativeai`)
- **LangChain** (HuggingFaceEmbeddings, FAISS vector store, ConversationalRetrievalChain)
- **Streamlit** for a ChatGPT-like user interface

## Features
- **File Upload**: Upload a local `.txt` file to form a document knowledge base.
- **Vector Store**: Build a FAISS index from the uploaded documents using HuggingFace embeddings.
- **Conversational Retrieval**: Query the knowledge base via a conversational chain, augmented by Google’s LLM.
- **Chat Interface**: Uses Streamlit’s `st.chat_input` and `st.chat_message` for a ChatGPT-like experience.

## Getting Started
1. **Install dependencies** (sample):
   ```bash
   pip install -r requirements.txt
   ```
   Make sure you have valid keys for Google Generative AI in your environment.

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

3. **Usage**:
   - Open the Streamlit app in your browser.
   - Click **"Upload a text file"** on the sidebar and select a `.txt` file.
   - Type your question in the **chat box** at the bottom and press **Enter**.
   - The app will generate a response using the uploaded text as context.

## How It Works
1. **TextLoader** loads your `.txt` file into documents.
2. **HuggingFaceEmbeddings** converts the documents into vector representations.
3. **FAISS** stores these vectors for fast similarity search.
4. **ConversationalRetrievalChain** retrieves context-relevant text for each query, passing it to Google’s LLM (Gemini) for a final answer.
