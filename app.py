import streamlit as st
import os
import warnings
import time
import json
import requests
import tempfile
from datetime import datetime
warnings.filterwarnings("ignore")

import tiktoken
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

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
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
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
    .knowledge-source {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🩺 MedAssist AI</h1>
    <p>Advanced Medical Question Answering System powered by FAISS & Medical Knowledge</p>
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

class MedicalAIClient:
    def __init__(self, api_key, use_openai_embeddings=False):
        self.api_key = api_key
        self.use_openai_embeddings = use_openai_embeddings
    
    def is_configured(self):
        return bool(self.api_key and self.api_key.startswith('sk-'))
    
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

@st.cache_data
def load_comprehensive_medical_knowledge():
    """Load comprehensive medical knowledge base"""
    medical_docs = [
        # Cardiovascular Conditions
        "Myocardial infarction (heart attack) presents with chest pain that may radiate to the left arm, jaw, or back. Associated symptoms include shortness of breath, nausea, sweating, and anxiety. ST-elevation MI (STEMI) requires immediate primary PCI or thrombolytic therapy within 90 minutes. Non-ST elevation MI (NSTEMI) is managed with antiplatelet therapy, anticoagulation, and risk stratification. Key medications include aspirin, clopidogrel, atorvastatin, metoprolol, and ACE inhibitors. Complications include arrhythmias, heart failure, and mechanical complications.",
        
        "Hypertension is defined as systolic BP ≥140 mmHg or diastolic BP ≥90 mmHg on repeated measurements. Stage 1 hypertension is 130-139/80-89 mmHg. First-line treatments include ACE inhibitors (lisinopril), ARBs (losartan), thiazide diuretics (hydrochlorothiazide), and calcium channel blockers (amlodipine). Lifestyle modifications include sodium restriction (<2.3g/day), weight loss, regular exercise, and alcohol moderation. Target BP is <130/80 mmHg for most patients.",
        
        "Heart failure with reduced ejection fraction (HFrEF) is treated with ACE inhibitors or ARBs, beta-blockers (metoprolol, carvedilol), and mineralocorticoid receptor antagonists (spironolactone). Newer therapies include SGLT2 inhibitors (dapagliflozin) and ARNI (sacubitril/valsartan). Diuretics manage volume overload. Symptoms include dyspnea on exertion, orthopnea, paroxysmal nocturnal dyspnea, and peripheral edema. NYHA classification grades functional capacity from I to IV.",
        
        # Endocrine Disorders
        "Type 2 diabetes mellitus is diagnosed with fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, or random glucose ≥200 mg/dL with symptoms. Metformin is first-line therapy unless contraindicated. Second-line options include sulfonylureas, DPP-4 inhibitors, GLP-1 agonists, SGLT2 inhibitors, and insulin. Target HbA1c is <7% for most adults. Complications include diabetic nephropathy, retinopathy, neuropathy, and accelerated cardiovascular disease. Annual screening includes eye exams, kidney function, and foot examinations.",
        
        "Diabetic ketoacidosis (DKA) presents with hyperglycemia >250 mg/dL, ketosis, and metabolic acidosis. Symptoms include polyuria, polydipsia, nausea, vomiting, and altered mental status. Treatment includes IV fluid resuscitation, insulin infusion, electrolyte replacement (especially potassium), and correction of precipitating factors. Common triggers include infection, medication non-compliance, and new-onset diabetes. Monitor for complications including cerebral edema in children.",
        
        "Thyroid disorders: Hyperthyroidism presents with weight loss, palpitations, heat intolerance, and tremor. Graves' disease is the most common cause. Treatment includes anti-thyroid medications (methimazole, propylthiouracil), radioactive iodine, or surgery. Hypothyroidism presents with fatigue, weight gain, cold intolerance, and bradycardia. Treatment is levothyroxine replacement with TSH monitoring every 6-8 weeks until stable.",
        
        # Respiratory Conditions
        "Community-acquired pneumonia (CAP) presents with fever, productive cough, pleuritic chest pain, and dyspnea. CURB-65 score helps determine severity and treatment setting. Outpatient treatment includes amoxicillin or azithromycin. Hospitalized patients receive ceftriaxone plus azithromycin or respiratory fluoroquinolone (levofloxacin). Chest X-ray shows consolidation. Complications include pleural effusion, empyema, and respiratory failure.",
        
        "Asthma exacerbation presents with wheezing, shortness of breath, chest tightness, and coughing. Peak flow <50% of personal best indicates severe exacerbation. Treatment includes oxygen, bronchodilators (albuterol), corticosteroids (prednisone or methylprednisolone), and magnesium sulfate for severe cases. Controller medications include inhaled corticosteroids (fluticasone), long-acting beta-agonists (salmeterol), and leukotriene inhibitors (montelukast).",
        
        "Chronic obstructive pulmonary disease (COPD) is characterized by airflow limitation due to emphysema and chronic bronchitis. Smoking cessation is the most important intervention. Bronchodilators include short-acting (albuterol) and long-acting (tiotropium) agents. Inhaled corticosteroids are added for frequent exacerbations. Oxygen therapy is indicated for severe hypoxemia. Exacerbations are treated with bronchodilators, corticosteroids, and antibiotics if bacterial infection is suspected.",
        
        # Infectious Diseases
        "Sepsis is life-threatening organ dysfunction due to dysregulated host response to infection. qSOFA score includes altered mental status, systolic BP ≤100 mmHg, and respiratory rate ≥22/min. Treatment follows the sepsis bundle: obtain blood cultures, administer broad-spectrum antibiotics within 1 hour, and provide IV fluid resuscitation. Vasopressors (norepinephrine) are used for shock. Source control is essential. Procalcitonin may guide antibiotic duration.",
        
        "Urinary tract infection (UTI) presents with dysuria, frequency, urgency, and suprapubic pain. Uncomplicated cystitis in women is treated with nitrofurantoin, trimethoprim-sulfamethoxazole, or fosfomycin. Complicated UTIs and pyelonephritis require fluoroquinolones or cephalosporins. Urine culture is indicated for recurrent infections, treatment failures, or complicated cases. Pregnant women require treatment even for asymptomatic bacteriuria.",
        
        "Antibiotic selection: Penicillins (amoxicillin) for streptococcal infections, cephalosporins (cephalexin) for skin and soft tissue, fluoroquinolones (ciprofloxacin) for gram-negative infections, macrolides (azithromycin) for atypical pathogens, and vancomycin for MRSA. Beta-lactam allergies require alternative agents. C. difficile colitis is a serious complication of antibiotic use requiring metronidazole or vancomycin.",
        
        # Emergency Medicine
        "Anaphylaxis is a severe allergic reaction requiring immediate epinephrine 0.3-0.5mg IM in the anterolateral thigh. Symptoms include difficulty breathing, facial/throat swelling, urticaria, gastrointestinal symptoms, and cardiovascular collapse. Additional treatments include H1 antihistamines (diphenhydramine), H2 blockers (ranitidine), corticosteroids (methylprednisolone), and bronchodilators. Biphasic reactions can occur 4-12 hours later. Common triggers include foods (nuts, shellfish), medications (penicillin), and insect stings.",
        
        "Acute stroke symptoms follow FAST assessment: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services. CT scan differentiates ischemic from hemorrhagic stroke. Ischemic stroke treatment includes IV tPA within 4.5 hours if no contraindications, and mechanical thrombectomy within 24 hours for large vessel occlusion. Blood pressure management is crucial - avoid aggressive reduction in acute ischemic stroke.",
        
        "Acute coronary syndrome (ACS) includes STEMI, NSTEMI, and unstable angina. Initial management includes aspirin, clopidogrel, atorvastatin, metoprolol, and anticoagulation with heparin. STEMI requires primary PCI within 90 minutes or fibrinolytic therapy within 30 minutes if PCI unavailable. NSTEMI is managed with risk stratification using TIMI or GRACE scores. Troponin levels help diagnose myocardial injury.",
        
        # Gastroenterology
        "Gastroesophageal reflux disease (GERD) presents with heartburn, regurgitation, and chest pain. Complications include Barrett's esophagus and adenocarcinoma. Proton pump inhibitors (omeprazole, pantoprazole) are first-line therapy. H2 receptor blockers (ranitidine) are less effective. Lifestyle modifications include weight loss, elevation of head of bed, and avoiding trigger foods. Endoscopy is indicated for alarm symptoms or failed medical therapy.",
        
        "Peptic ulcer disease is caused by H. pylori infection or NSAIDs. Triple therapy for H. pylori includes PPI + clarithromycin + amoxicillin for 14 days. Quadruple therapy adds metronidazole. NSAID-induced ulcers are treated with PPIs and NSAID discontinuation. Bleeding ulcers may require endoscopic intervention. Complications include perforation and gastric outlet obstruction.",
        
        # Pharmacology
        "Metformin is first-line therapy for type 2 diabetes with multiple benefits including weight neutrality and cardiovascular protection. Contraindications include severe kidney disease (eGFR <30), liver disease, heart failure, and conditions predisposing to lactic acidosis. Common side effects include gastrointestinal upset and vitamin B12 deficiency. Dose adjustment is required for eGFR 30-45 mL/min/1.73m². Maximum dose is 2550mg daily divided with meals.",
        
        "ACE inhibitors (lisinopril, enalapril) are first-line for hypertension and heart failure. Benefits include renal protection in diabetes and post-MI mortality reduction. Side effects include dry cough (10-15%), hyperkalemia, and angioedema (rare but serious). ARBs (losartan, valsartan) have similar efficacy with lower cough incidence. Monitor kidney function and potassium levels. Contraindicated in pregnancy.",
        
        "Warfarin is a vitamin K antagonist requiring INR monitoring. Target INR is 2-3 for most indications, 2.5-3.5 for mechanical heart valves. Drug interactions are numerous, especially with antibiotics and antifungals. Dietary vitamin K intake should be consistent. Reversal agents include vitamin K, fresh frozen plasma, and prothrombin complex concentrate. Novel oral anticoagulants (DOACs) like rivaroxaban require less monitoring.",
        
        # Mental Health
        "Major depressive disorder is diagnosed with ≥5 symptoms for ≥2 weeks including depressed mood or anhedonia. SSRIs (sertraline, escitalopram) are first-line therapy with 4-6 week trial periods. SNRIs (venlafaxine) are alternatives. Suicide risk assessment is essential. Psychotherapy, particularly CBT, is equally effective. Combination therapy may be superior for severe depression. Monitor for activation symptoms in young adults.",
        
        "Anxiety disorders include generalized anxiety disorder, panic disorder, and social anxiety. SSRIs and SNRIs are first-line treatments. Benzodiazepines (lorazepam, alprazolam) provide rapid relief but have addiction potential. CBT and exposure therapy are effective non-pharmacological treatments. Beta-blockers (propranolol) help with performance anxiety. Avoid alcohol and caffeine which can worsen symptoms."
    ]
    
    return medical_docs

@st.cache_resource
def setup_medical_vectorstore(use_openai_embeddings=False, api_key=None):
    """Setup medical knowledge base with FAISS"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📄 Loading comprehensive medical knowledge...")
        progress_bar.progress(20)
        
        # Load medical documents
        medical_docs_text = load_comprehensive_medical_knowledge()
        
        # Convert to Document objects
        documents = [Document(page_content=doc, metadata={"source": f"medical_knowledge_{i}"}) 
                    for i, doc in enumerate(medical_docs_text)]
        
        progress_bar.progress(40)
        status_text.text("✂️ Processing medical content...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        
        document_chunks = text_splitter.split_documents(documents)
        
        progress_bar.progress(60)
        status_text.text("🧮 Creating medical embeddings...")
        
        # Choose embedding model
        if use_openai_embeddings and api_key:
            embedding_model = OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-ada-002"
            )
        else:
            embedding_model = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        
        progress_bar.progress(80)
        status_text.text("🗄️ Building FAISS vector store...")
        
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=document_chunks,
            embedding=embedding_model
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 4}
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Medical knowledge base ready!")
        
        # Store stats
        st.session_state.total_chunks = len(document_chunks)
        st.session_state.total_topics = len(medical_docs_text)
        st.session_state.embedding_model = "OpenAI" if use_openai_embeddings else "SentenceTransformer"
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return retriever
        
    except Exception as e:
        st.error(f"❌ Failed to setup knowledge base: {str(e)}")
        return None

def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def generate_medical_rag_response(ai_client, retriever, user_input, model="gpt-3.5-turbo", max_tokens=512):
    """Generate RAG response using FAISS vector search"""
    system_message = """You are MedAssist AI, an advanced medical AI assistant with access to comprehensive medical knowledge.

GUIDELINES:
- Answer based ONLY on the provided medical context from your knowledge base
- Structure responses clearly with relevant medical sections when appropriate
- Use appropriate medical terminology while remaining accessible to healthcare professionals
- If the context lacks specific information, acknowledge this limitation clearly
- For treatment questions, present evidence-based options systematically
- Include relevant dosages, contraindications, and monitoring parameters when discussing medications
- Always emphasize consulting with healthcare professionals for diagnosis and treatment decisions
- Provide differential diagnoses when relevant

RESPONSE FORMAT:
- Use clear headings when appropriate (## Symptoms, ## Diagnosis, ## Treatment, ## Monitoring)
- Use bullet points for lists of symptoms, treatments, or differential diagnoses
- **Bold** key medical terms, conditions, and medications
- Include specific dosages and clinical guidelines when available"""

    try:
        # Retrieve relevant documents from FAISS
        relevant_docs = retriever.get_relevant_documents(query=user_input)
        
        if not relevant_docs:
            return "⚠️ No relevant medical information found in the knowledge base for this query."
        
        # Combine context from retrieved documents
        context_list = [doc.page_content for doc in relevant_docs]
        context = "\n\n---\n\n".join(context_list)
        
        # Ensure context fits within token limits
        max_context_tokens = 2000
        if count_tokens(context) > max_context_tokens:
            # Truncate context while preserving complete sentences
            sentences = context.split('. ')
            truncated_context = ""
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence + '. ')
                if current_tokens + sentence_tokens <= max_context_tokens:
                    truncated_context += sentence + '. '
                    current_tokens += sentence_tokens
                else:
                    break
            context = truncated_context
        
        prompt = f"""{system_message}

MEDICAL CONTEXT FROM KNOWLEDGE BASE:
{context}

HEALTHCARE PROFESSIONAL QUESTION:
{user_input}

MEDICAL RESPONSE (based only on the provided context):"""
        
        # Generate response using AI
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
        st.header("🔑 AI Configuration")
        
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
        
        # Embedding model choice
        use_openai_embeddings = st.checkbox(
            "Use OpenAI Embeddings",
            value=False,
            help="More accurate but uses API credits. Unchecked uses free SentenceTransformers."
        )
        
        model = st.selectbox(
            "AI Model", 
            ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"],
            help="gpt-3.5-turbo is fastest and most cost-effective"
        )
        
        if api_key:
            embedding_info = "OpenAI Embeddings" if use_openai_embeddings else "SentenceTransformers (Free)"
            st.info(f"🤖 Model: {model}")
            st.info(f"📊 Embeddings: {embedding_info}")
        else:
            st.warning("⚠️ Please add your OpenAI API key")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        max_tokens = st.slider("Response Length", 256, 1024, 512)
    
    # Initialize AI client
    ai_client = MedicalAIClient(api_key, use_openai_embeddings)
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ai_client.is_configured():
            st.markdown('<div class="status-indicator status-success">🟢 AI Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-warning">🟡 API Key Needed</div>', unsafe_allow_html=True)
    
    with col2:
        session_duration = datetime.now() - st.session_state.session_start
        st.markdown(f'<div class="status-indicator">⏱️ {str(session_duration).split(".")[0]}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="status-indicator">💬 Queries: {st.session_state.query_count}</div>', unsafe_allow_html=True)
    
    # Initialize knowledge base
    if not st.session_state.system_initialized:
        if not api_key and use_openai_embeddings:
            st.warning("⚠️ OpenAI API key required for OpenAI embeddings. Using SentenceTransformers instead.")
            use_openai_embeddings = False
        
        with st.spinner("🚀 Initializing medical knowledge base with FAISS..."):
            retriever = setup_medical_vectorstore(use_openai_embeddings, api_key)
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.system_initialized = True
                st.success("✅ MedAssist AI is ready for medical consultations!")
                st.rerun()
    
    # Sidebar info
    with st.sidebar:
        if hasattr(st.session_state, 'total_chunks'):
            st.header("📊 Knowledge Base")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Topics", st.session_state.total_topics)
            with col_b:
                st.metric("Chunks", st.session_state.total_chunks)
            
            if hasattr(st.session_state, 'embedding_model'):
                st.info(f"🔬 Using: {st.session_state.embedding_model}")
        
        st.header("🩺 Sample Medical Questions")
        sample_questions = [
            "What are the clinical signs and symptoms of acute myocardial infarction?",
            "How is Type 2 diabetes mellitus diagnosed and managed?",
            "What are the contraindications and side effects of Metformin?",
            "How does chronic hypertension affect cardiovascular and renal systems?",
            "What is the emergency management protocol for anaphylaxis?",
            "What are the differential diagnoses for acute chest pain?",
            "How is community-acquired pneumonia diagnosed and treated?",
            "What are the key features of diabetic ketoacidosis?",
            "How do ACE inhibitors work and what are their side effects?",
            "What is the approach to managing acute asthma exacerbation?"
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
        
        with st.spinner("🔍 Consulting medical knowledge base with FAISS..."):
            response = generate_medical_rag_response(
                ai_client, 
                st.session_state.retriever, 
                user_input, 
                model=model, 
                max_tokens=max_tokens
            )
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Show knowledge sources if response exists
    if st.session_state.messages and st.session_state.system_initialized:
        with st.expander("🔍 View Knowledge Sources (Last Query)"):
            if st.session_state.messages:
                last_question = [msg for msg in st.session_state.messages if msg["role"] == "user"]
                if last_question:
                    docs = st.session_state.retriever.get_relevant_documents(last_question[-1]["content"])
                    for i, doc in enumerate(docs):
                        st.markdown(f"""
                        <div class="knowledge-source">
                            <strong>Source {i+1}:</strong><br>
                            {doc.page_content[:400]}...
                        </div>
                        """, unsafe_allow_html=True)
    
    # Medical disclaimer
    st.markdown("""
    <div class="medical-disclaimer">
        <strong>⚠️ Important Medical Disclaimer</strong><br>
        This AI assistant provides information for educational purposes only and should not replace professional medical advice, diagnosis, or treatment. 
        The responses are based on medical knowledge and should be verified against current medical literature and clinical guidelines.
        Always consult with qualified healthcare professionals for medical decisions. In case of medical emergencies, contact emergency services immediately.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()