import streamlit as st
import os
import warnings
import time
import json
import requests
from datetime import datetime
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

import tiktoken
import pandas as pd

st.set_page_config(
    page_title="MedAssist AI - Medical RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

st.markdown("""
<div class="main-header">
    <h1>🩺 MedAssist AI</h1>
    <p>Advanced Medical Question Answering System powered by AI & Medical Knowledge</p>
</div>
""", unsafe_allow_html=True)

if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()

class OpenAIClient:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def is_configured(self):
        return bool(self.api_key and self.api_key.startswith('sk-'))
    
    def get_embeddings(self, texts):
        """Get embeddings for texts using OpenAI"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "text-embedding-ada-002",
                "input": texts
            }
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                embeddings = [item['embedding'] for item in response.json()['data']]
                return np.array(embeddings)
            else:
                return None
                
        except Exception as e:
            st.error(f"Embedding error: {e}")
            return None
    
    def generate(self, prompt, model="gpt-3.5-turbo", max_tokens=512):
        if not self.is_configured():
            return "❌ OpenAI API key not configured"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
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
            elif response.status_code == 401:
                return "❌ Invalid API key. Please check your OpenAI API key."
            elif response.status_code == 429:
                return "❌ Rate limit exceeded. Please try again in a moment."
            else:
                return f"❌ OpenAI API Error: {response.status_code}"
                
        except Exception as e:
            return f"❌ Connection Error: {str(e)}"

class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        
    def add_documents(self, texts, embeddings):
        self.documents = texts
        self.embeddings = embeddings
    
    def similarity_search(self, query_embedding, k=3):
        if self.embeddings is None:
            return []
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return top documents
        return [self.documents[i] for i in top_indices]

@st.cache_data
def load_medical_knowledge():
    """Load medical knowledge base"""
    # Comprehensive medical knowledge base
    medical_docs = [
        "Myocardial infarction (heart attack) presents with chest pain, shortness of breath, nausea, sweating, and radiating pain to left arm or jaw. Immediate treatment includes aspirin, oxygen, nitroglycerin, and emergency cardiac catheterization. Risk factors include smoking, diabetes, hypertension, and family history.",
        
        "Type 2 diabetes mellitus is diagnosed with fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, or random glucose ≥200 mg/dL with symptoms. Treatment includes lifestyle modifications, metformin as first-line medication, and progression to insulin if needed. Complications include nephropathy, retinopathy, and neuropathy.",
        
        "Pneumonia symptoms include fever, productive cough, chest pain, shortness of breath, and fatigue. Community-acquired pneumonia is typically treated with amoxicillin or macrolides. Hospital-acquired pneumonia requires broader spectrum antibiotics. Chest X-ray shows consolidation.",
        
        "Anaphylaxis is a severe allergic reaction requiring immediate epinephrine administration. Symptoms include difficulty breathing, swelling of face/throat, rapid pulse, dizziness, and full-body rash. Common triggers include foods (nuts, shellfish), medications (penicillin), and insect stings.",
        
        "Hypertension is defined as systolic BP ≥140 mmHg or diastolic BP ≥90 mmHg on repeated measurements. Treatment includes ACE inhibitors, ARBs, thiazide diuretics, and calcium channel blockers. Lifestyle modifications include salt restriction, weight loss, and exercise.",
        
        "Stroke symptoms follow FAST protocol: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services. Ischemic stroke treatment includes tPA within 4.5 hours and thrombectomy within 24 hours. Hemorrhagic stroke requires blood pressure control and neurosurgical evaluation.",
        
        "Appendicitis presents with periumbilical pain migrating to right lower quadrant, nausea, vomiting, and fever. McBurney's point tenderness is classic. Treatment is appendectomy, either laparoscopic or open. Complications include perforation and abscess formation.",
        
        "Asthma exacerbation symptoms include wheezing, shortness of breath, chest tightness, and coughing. Treatment includes bronchodilators (albuterol), corticosteroids, and oxygen. Severe cases may require epinephrine and mechanical ventilation.",
        
        "Metformin is first-line treatment for type 2 diabetes. Contraindications include kidney disease (eGFR <30), liver disease, and conditions predisposing to lactic acidosis. Side effects include gastrointestinal upset and rare lactic acidosis. Dose adjustment needed in renal impairment.",
        
        "Sepsis is defined as life-threatening organ dysfunction due to dysregulated host response to infection. Early recognition and treatment within 1 hour improves outcomes. Treatment includes broad-spectrum antibiotics, fluid resuscitation, and vasopressors if needed.",
        
        "Heart failure symptoms include shortness of breath, fatigue, ankle swelling, and orthopnea. New York Heart Association (NYHA) classification grades functional capacity. Treatment includes ACE inhibitors, beta-blockers, diuretics, and lifestyle modifications.",
        
        "Chronic obstructive pulmonary disease (COPD) is characterized by airflow limitation due to smoking. Symptoms include chronic cough, sputum production, and dyspnea. Treatment includes bronchodilators, inhaled corticosteroids, and smoking cessation.",
        
        "Gastroesophageal reflux disease (GERD) presents with heartburn, regurgitation, and chest pain. Complications include Barrett's esophagus and adenocarcinoma. Treatment includes proton pump inhibitors, H2 blockers, and lifestyle modifications.",
        
        "Urinary tract infection (UTI) symptoms include dysuria, frequency, urgency, and suprapubic pain. Diagnosis requires urinalysis and culture. Treatment includes trimethoprim-sulfamethoxazole, nitrofurantoin, or fluoroquinolones.",
        
        "Migraine headaches are characterized by unilateral, throbbing pain with photophobia, phonophobia, and nausea. Triggers include stress, hormonal changes, and certain foods. Treatment includes triptans for acute episodes and preventive medications for frequent attacks."
    ]
    
    return medical_docs

def setup_knowledge_base(ai_client):
    """Setup medical knowledge base with embeddings"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📄 Loading medical knowledge...")
        progress_bar.progress(20)
        
        # Load medical documents
        medical_docs = load_medical_knowledge()
        
        status_text.text("🧮 Creating embeddings...")
        progress_bar.progress(50)
        
        # Get embeddings from OpenAI
        embeddings = ai_client.get_embeddings(medical_docs)
        
        if embeddings is None:
            st.error("Failed to create embeddings. Please check your API key.")
            return None
        
        progress_bar.progress(80)
        status_text.text("🗄️ Building vector store...")
        
        # Create simple vector store
        vector_store = SimpleVectorStore()
        vector_store.add_documents(medical_docs, embeddings)
        
        progress_bar.progress(100)
        status_text.text("✅ Medical knowledge base ready!")
        
        # Store stats
        st.session_state.total_chunks = len(medical_docs)
        st.session_state.total_sources = "15 medical topics"
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return vector_store
        
    except Exception as e:
        st.error(f"❌ Failed to setup knowledge base: {str(e)}")
        return None

def generate_rag_response(ai_client, vector_store, user_input, model="gpt-3.5-turbo", max_tokens=512):
    """Generate RAG response using simple vector search"""
    system_message = """You are MedAssist AI, a medical assistant with access to medical knowledge.

GUIDELINES:
- Answer based ONLY on the provided medical context
- Structure responses clearly with relevant medical sections
- Use appropriate medical terminology while remaining accessible
- If the context lacks specific information, acknowledge this limitation clearly
- For treatment questions, present options systematically
- Always emphasize consulting with healthcare professionals

RESPONSE FORMAT:
- Use clear headings when appropriate (## Symptoms, ## Diagnosis, ## Treatment)
- Use bullet points for lists of symptoms, treatments, or differential diagnoses
- **Bold** key medical terms and conditions"""

    try:
        # Get query embedding
        query_embedding = ai_client.get_embeddings([user_input])
        
        if query_embedding is None:
            return "❌ Failed to process query. Please try again."
        
        # Search for relevant documents
        relevant_docs = vector_store.similarity_search(query_embedding[0], k=3)
        
        if not relevant_docs:
            return "⚠️ No relevant medical information found for this query."
        
        context = "\n\n---\n\n".join(relevant_docs)
        
        prompt = f"""{system_message}

MEDICAL CONTEXT:
{context}

QUESTION: {user_input}

MEDICAL RESPONSE (based only on the provided context):"""
        
        response = ai_client.generate(prompt, model=model, max_tokens=max_tokens)
        
        if not response.startswith("❌"):
            response += "\n\n---\n*Response based on medical knowledge base. Always verify with current medical literature and clinical guidelines.*"
        
        return response
        
    except Exception as e:
        return f'❌ Error generating medical response: {str(e)}'

def main():
    # API Configuration in sidebar
    with st.sidebar:
        st.markdown('<div class="api-config">', unsafe_allow_html=True)
        st.header("🔑 OpenAI Configuration")
        
        # Check for API key in environment first
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                help="Get your API key from https://platform.openai.com/api-keys"
            )
        else:
            st.success("✅ API key loaded from environment")
        
        model = st.selectbox(
            "Model", 
            ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"],
            help="gpt-3.5-turbo is fastest and cheapest"
        )
        
        if api_key:
            st.info(f"🤖 Using: {model}")
        else:
            st.warning("⚠️ Please add your OpenAI API key")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        max_tokens = st.slider("Response Length", 256, 1024, 512)
    
    # Initialize AI client
    ai_client = OpenAIClient(api_key)
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ai_client.is_configured():
            st.markdown('<div class="status-indicator status-success">🟢 OpenAI Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-warning">🟡 API Key Needed</div>', unsafe_allow_html=True)
    
    with col2:
        session_duration = datetime.now() - st.session_state.session_start
        st.markdown(f'<div class="status-indicator">⏱️ {str(session_duration).split(".")[0]}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="status-indicator">💬 Queries: {st.session_state.query_count}</div>', unsafe_allow_html=True)
    
    # Initialize knowledge base
    if not st.session_state.system_initialized and api_key:
        with st.spinner("🚀 Initializing medical knowledge base..."):
            vector_store = setup_knowledge_base(ai_client)
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.system_initialized = True
                st.success("✅ MedAssist AI is ready!")
                st.rerun()
    
    # Sidebar info
    with st.sidebar:
        if hasattr(st.session_state, 'total_chunks'):
            st.header("📊 Knowledge Base")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Topics", 15)
            with col_b:
                st.metric("Chunks", st.session_state.total_chunks)
        
        st.header("🩺 Sample Questions")
        sample_questions = [
            "What are the symptoms of myocardial infarction?",
            "How is Type 2 diabetes diagnosed?",
            "What are the signs of pneumonia?",
            "What is the treatment for anaphylaxis?",
            "How is hypertension defined?",
            "What are the symptoms of stroke?",
            "How is appendicitis diagnosed?",
            "What are the side effects of Metformin?"
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
                <strong>👩‍⚕️ Question:</strong><br>
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
    user_input = st.chat_input("💬 Ask a medical question...")
    
    # Handle sample question selection
    if hasattr(st.session_state, 'user_input') and st.session_state.user_input:
        user_input = st.session_state.user_input
        del st.session_state.user_input
    
    if user_input and st.session_state.system_initialized:
        if not ai_client.is_configured():
            st.error("Please configure your OpenAI API key first!")
            return
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.query_count += 1
        
        with st.spinner("🔍 Consulting medical AI..."):
            response = generate_rag_response(
                ai_client, 
                st.session_state.vector_store, 
                user_input, 
                model=model, 
                max_tokens=max_tokens
            )
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Medical disclaimer
    st.markdown("""
    <div class="medical-disclaimer">
        <strong>⚠️ Medical Disclaimer</strong><br>
        This AI provides educational information only. Always consult healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()