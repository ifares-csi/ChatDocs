# WebChat: Chat with Web Pages using DeepSeek R1

## Overview
WebChat is a Streamlit-based web application that allows users to interact with web pages by summarizing their content and answering user queries using the Deepseek R1 language model. The app extracts textual content from a given URL, processes it into vector embeddings using FAISS, and retrieves relevant responses based on the user prompt.


## Features
- Extracts content from any webpage URL
- Summarizes webpage content using Deepseek R1 (1.5B model)
- Stores vector embeddings with FAISS for efficient retrieval
- Provides a structured reasoning process with a final answer
- User-friendly UI built with Streamlit

## Installation

### Prerequisites
- Python 3.12
- deepSeek R1
- Streamlit
- LangChain
- FAISS
- Ollama
- Requests
- Dotenv

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ifares-csi/WebChat.git
   cd WebChat
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file and add:
   ```bash
   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGSMITH_PROJECT=your_project_name
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Enter a valid webpage URL.
2. Provide a question or request a summary.
3. Click the **Submit** button.
4. The application will extract content, process it, and generate a response.

## Technologies Used
- **DeepSeek R1 (1.5B model)**: for response generation and reasoning
- **Streamlit**: UI framework
- **Ollama**: LLM model provider
- **LangChain**: AI framework for document retrieval
- **FAISS**: Vector database for efficient storage and retrieval
- **Requests**: Webpage content extraction
- **Dotenv**: Environment variable management

## License
This project is licensed under the MIT License.

## Author
**El-Fares**

## Contributions
Contributions are welcome! Feel free to fork the repository and submit pull requests.
