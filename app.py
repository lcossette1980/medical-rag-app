import streamlit as st
import os
import warnings
import time
import json
import requests
from datetime import datetime
warnings.filterwarnings("ignore")

import tiktoken
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(
    page_title="MedAssist AI - Medical RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for medical theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        margin: 0.25rem;
    }
    .status-success {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .status-warning {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .status-error {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .user-message {
        background: #f1f5f9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .assistant-message {
        background: #f0fdf4;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #059669;
        margin: 1rem 0;
    }
    .medical-disclaimer {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #991b1b;
    }
    .api-config {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🩺 MedAssist AI</h1>
    <p>Advanced Medical Question Answering System powered by AI & Merck Manual</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()

# Cloud AI Client (supports multiple providers)
class CloudAIClient:
    def __init__(self):
        self.provider = None
        self.api_key = None
        self.model = None
    
    def configure(self, provider, api_key, model):
        self.provider = provider
        self.api_key = api_key
        self.model = model
    
    def is_configured(self):
        return self.provider and self.api_key and self.model
    
    def generate_openai(self, prompt, max_tokens=512):
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                return f"❌ OpenAI API Error: {response.status_code}"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate_anthropic(self, prompt, max_tokens=512):
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"].strip()
            else:
                return f"❌ Anthropic API Error: {response.status_code}"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate_ollama_cloud(self, prompt, max_tokens=512):
        """For services like Replicate, Together AI, etc."""
        try:
            # Example for Replicate
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "version": self.model,
                "input": {
                    "prompt": prompt,
                    "max_length": max_tokens,
                    "temperature": 0
                }
            }
            
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 201:
                # Handle Replicate's async response
                prediction_url = response.json()["urls"]["get"]
                
                # Poll for completion
                for _ in range(30):  # Wait up to 30 seconds
                    time.sleep(1)
                    result_response = requests.get(prediction_url, headers=headers)
                    result = result_response.json()
                    
                    if result["status"] == "succeeded":
                        return "".join(result["output"])
                    elif result["status"] == "failed":
                        return f"❌ Generation failed: {result.get('error', 'Unknown error')}"
                
                return "❌ Request timed out"
            else:
                return f"❌ API Error: {response.status_code}"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate(self, prompt, max_tokens=512):
        if not self.is_configured():
            return "❌ AI provider not configured"
        
        if self.provider == "openai":
            return self.generate_openai(prompt, max_tokens)
        elif self.provider == "anthropic":
            return self.generate_anthropic(prompt, max_tokens)
        elif self.provider == "replicate":
            return self.generate_ollama_cloud(prompt, max_tokens)
        else:
            return f"❌ Unsupported provider: {self.provider}"

@st.cache_resource
def setup_vectorstore():
    """Setup the vector database"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📄 Loading medical documentation...")
        progress_bar.progress(20)
        
        manual_pdf_path = "medical_diagnosis_manual.pdf"
        if not os.path.exists(manual_pdf_path):
            st.error("❌ Medical manual PDF not found. Please add 'medical_diagnosis_manual.pdf' to the app directory.")
            st.info("💡 For demo purposes, you can upload any medical PDF and rename it.")
            st.stop()
        
        pdf_loader = PyMuPDFLoader(manual_pdf_path)
        progress_bar.progress(40)
        
        status_text.text("✂️ Processing medical content...")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name='cl100k_base',
            chunk_size=1000,
            chunk_overlap=100
        )
        
        document_chunks = pdf_loader.load_and_split(text_splitter)
        progress_bar.progress(60)
        
        status_text.text("🧮 Creating medical embeddings...")
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        progress_bar.progress(80)
        
        status_text.text("🗄️ Building knowledge base...")
        out_dir = 'medical_db'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        vectorstore = Chroma.from_documents(
            document_chunks,
            embedding_model,
            persist_directory=out_dir
        )
        
        retriever = vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 3}
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Medical knowledge base ready!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Store document stats
        st.session_state.total_chunks = len(document_chunks)
        st.session_state.total_pages = len(pdf_loader.load())
        
        return retriever
        
    except Exception as e:
        st.error(f"❌ Failed to setup knowledge base: {str(e)}")
        return None

def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def generate_rag_response(ai_client, retriever, user_input, max_tokens=512):
    """Generate RAG response using cloud AI"""
    system_message = """You are MedAssist AI, a medical assistant with access to the Merck Manual.

GUIDELINES:
- Answer based ONLY on the provided medical context from the Merck Manual
- Structure responses clearly with relevant medical sections
- Use appropriate medical terminology while remaining accessible to healthcare professionals
- If the context lacks specific information, acknowledge this limitation clearly
- For treatment questions, present options systematically
- Always emphasize consulting with healthcare professionals for diagnosis and treatment decisions

RESPONSE FORMAT:
- Use clear headings when appropriate (## Symptoms, ## Diagnosis, ## Treatment)
- Use bullet points for lists of symptoms, treatments, or differential diagnoses
- **Bold** key medical terms and conditions
- Provide clinical context when available"""

    try:
        # Retrieve relevant chunks
        relevant_docs = retriever.get_relevant_documents(query=user_input, k=3)
        
        if not relevant_docs:
            return "⚠️ No relevant medical information found in the knowledge base for this query."
        
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""{system_message}

MEDICAL CONTEXT FROM MERCK MANUAL:
{context}

HEALTHCARE PROFESSIONAL QUESTION:
{user_input}

MEDICAL RESPONSE (based only on the provided context):"""
        
        # Generate response using cloud AI
        response = ai_client.generate(prompt, max_tokens=max_tokens)
        
        if not response.startswith("❌"):
            response += "\n\n---\n*Response based on Merck Manual content. Always verify with current medical literature and clinical guidelines.*"
        
        return response
        
    except Exception as e:
        return f'❌ Error generating medical response: {str(e)}'

def main():
    # AI Configuration
    with st.sidebar:
        st.markdown('<div class="api-config">', unsafe_allow_html=True)
        st.header("🤖 AI Configuration")
        
        provider = st.selectbox(
            "AI Provider",
            ["openai", "anthropic", "replicate", "local-ollama"],
            help="Choose your AI provider"
        )
        
        if provider == "openai":
            api_key = st.text_input("OpenAI API Key", type="password", help="Get from https://platform.openai.com/api-keys")
            model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
        elif provider == "anthropic":
            api_key = st.text_input("Anthropic API Key", type="password", help="Get from https://console.anthropic.com/")
            model = st.selectbox("Model", ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"])
        elif provider == "replicate":
            api_key = st.text_input("Replicate API Key", type="password", help="Get from https://replicate.com/account/api-tokens")
            model = st.selectbox("Model", ["meta/llama-2-7b-chat", "mistralai/mistral-7b-instruct-v0.1"])
        else:
            api_key = None
            model = "llama2"
            st.info("Using local Ollama - make sure it's running!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Configuration
        st.header("⚙️ Settings")
        max_tokens = st.slider("Response Length", 256, 1024, 512)
    
    # Initialize AI client
    ai_client = CloudAIClient()
    
    if provider != "local-ollama":
        if api_key:
            ai_client.configure(provider, api_key, model)
        else:
            st.warning("Please provide API key to continue")
            st.stop()
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if provider == "local-ollama":
            # Check local Ollama
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    st.markdown('<div class="status-indicator status-success">🟢 Ollama Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-indicator status-error">🔴 Ollama Offline</div>', unsafe_allow_html=True)
            except:
                st.markdown('<div class="status-indicator status-error">🔴 Ollama Offline</div>', unsafe_allow_html=True)
        else:
            if ai_client.is_configured():
                st.markdown('<div class="status-indicator status-success">🟢 AI Connected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-indicator status-warning">🟡 API Key Needed</div>', unsafe_allow_html=True)
    
    with col2:
        session_duration = datetime.now() - st.session_state.session_start
        st.markdown(f'<div class="status-indicator">⏱️ Session: {str(session_duration).split(".")[0]}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="status-indicator">💬 Queries: {st.session_state.query_count}</div>', unsafe_allow_html=True)
    
    # Initialize knowledge base
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing medical knowledge base..."):
            retriever = setup_vectorstore()
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.system_initialized = True
                st.success("✅ MedAssist AI is ready for medical consultations!")
                st.rerun()
    
    # Sample questions in sidebar
    with st.sidebar:
        if hasattr(st.session_state, 'total_pages'):
            st.header("📊 Knowledge Base")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Pages", st.session_state.total_pages)
            with col_b:
                st.metric("Chunks", st.session_state.total_chunks)
        
        st.header("🩺 Sample Questions")
        sample_questions = [
            "What are the clinical signs of myocardial infarction?",
            "How is Type 2 diabetes diagnosed?",
            "What are the contraindications for Metformin?",
            "What is the emergency treatment for anaphylaxis?",
            "What are the symptoms of pneumonia?"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_{i}", use_container_width=True):
                st.session_state.user_input = question
    
    # Main chat interface
    st.markdown("## 💬 Medical Consultation")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>👩‍⚕️ Healthcare Professional:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🩺 MedAssist AI:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("💬 Ask a medical question based on the Merck Manual...")
    
    # Handle sample question selection
    if hasattr(st.session_state, 'user_input') and st.session_state.user_input:
        user_input = st.session_state.user_input
        del st.session_state.user_input
    
    if user_input and st.session_state.system_initialized:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.query_count += 1
        
        # Generate response
        with st.spinner("🔍 Consulting AI with medical knowledge base..."):
            if provider == "local-ollama":
                # Use local Ollama if available
                try:
                    from pathlib import Path
                    if Path("app.py").exists():  # Fallback to previous Ollama logic
                        response = "Local Ollama not properly configured for cloud deployment"
                    else:
                        response = "Please configure a cloud AI provider"
                except:
                    response = "Please configure a cloud AI provider for online deployment"
            else:
                response = generate_rag_response(
                    ai_client, 
                    st.session_state.retriever, 
                    user_input, 
                    max_tokens=max_tokens
                )
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Medical disclaimer
    st.markdown("""
    <div class="medical-disclaimer">
        <strong>⚠️ Important Medical Disclaimer</strong><br>
        This AI assistant provides information for educational purposes only and should not replace professional medical advice, diagnosis, or treatment. 
        The responses are based on the Merck Manual content and should be verified against current medical literature and clinical guidelines.
        Always consult with qualified healthcare professionals for medical decisions. In case of medical emergencies, contact emergency services immediately.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()