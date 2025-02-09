import os
import json
import streamlit as st
import google.generativeai as genai
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from pydantic import Field
from typing import Optional

# 1. SET PAGE CONFIG FIRST!
st.set_page_config(
    page_title="ChatDocs - Gemini-powered RAG Chat",
    page_icon="💬",
    layout="centered"
)

# Configuration
SESSION_FILE = "chat_session.json"
VECTORSTORE_DIR = "vectorstore"

# Custom LLM class (keep previous implementation)
class GoogleGenerativeAI(LLM):
    api_key: str = Field(..., description="Google AI API Key")
    model_name: str = Field(default="gemini-2.0-flash")
    generation_config: dict = Field(
        default_factory=lambda: {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        },
        description="Generation configuration"
    )

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Error: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "google_generativeai"

# Initialize session state with persistence
def initialize_session_state():
    # Load from files if they exist
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            saved_state = json.load(f)
            st.session_state.update(saved_state)
    else:
        # Default values
        st.session_state.update({
            'chat_history': [],
            'vectorstore': None,
            'processed': False,
            'current_file': None
        })

    # Load vectorstore if exists (with dangerous deserialization allowed)
    if os.path.exists(VECTORSTORE_DIR):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.load_local(
                VECTORSTORE_DIR,
                embeddings,
                allow_dangerous_deserialization=True  # Add this parameter
            )
            st.session_state.processed = True
        except Exception as e:
            st.error(f"Error loading vectorstore: {str(e)}")

def save_session_state():
    # Save chat history to file
    with open(SESSION_FILE, "w") as f:
        json.dump({
            'chat_history': st.session_state.chat_history,
            'processed': st.session_state.processed,
            'current_file': st.session_state.current_file
        }, f)

    # Save vectorstore if exists
    if st.session_state.vectorstore:
        st.session_state.vectorstore.save_local(VECTORSTORE_DIR)

# Initialize after page config
initialize_session_state()

# Sidebar for configurations
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Enter Google AI API Key:", type="password")
    uploaded_file = st.file_uploader("Upload knowledge base", type=["txt"])
    
    # Process file only when uploaded and not already processed
    if uploaded_file and api_key and not st.session_state.processed:
        with st.spinner("Processing document..."):
            try:
                # Create temporary file
                temp_file = f"./{uploaded_file.name}"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process document
                loader = TextLoader(temp_file)
                docs = loader.load()
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                st.session_state.processed = True  # Mark as processed
                st.toast("✅ Document processed successfully!", icon="✅")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.session_state.processed = False
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    if st.button("🧹 New Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
st.title("💬 ChatDocs")
st.caption("Powered by Google Gemini 2.0 Flash & LangChain RAG")

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("Ask about your document..."):
    if not api_key:
        st.error("🔑 Please enter your API key in the sidebar")
        st.stop()
    
    if not st.session_state.vectorstore:
        st.error("📄 Please upload a document first")
        st.stop()

    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            try:
                llm = GoogleGenerativeAI(api_key=api_key)
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever(),
                    return_source_documents=True
                )
                
                # Format chat history for LangChain
                lc_history = [
                    (msg["content"], next_msg["content"]) 
                    for msg, next_msg in zip(
                        st.session_state.chat_history[::2], 
                        st.session_state.chat_history[1::2]
                    )
                ]
                
                # Get response
                result = qa_chain({
                    "question": prompt,
                    "chat_history": lc_history
                })
                
                response = result["answer"]
                # if "source_documents" in result:
                #     response += "\n\n📚 Sources:\n" + "\n".join(
                #         [f"- {doc.metadata['source']}" for doc in result["source_documents"]]
                #     )
                
            except Exception as e:
                response = f"❌ Error: {str(e)}"

            # Display and store response
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    save_session_state()