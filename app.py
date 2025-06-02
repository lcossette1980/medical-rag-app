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

# Enhanced CSS with modern design, gradients, and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
        animation: slideInDown 0.8s ease-out;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 30s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 4px 8px rgba(0,0,0,0.2);
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.25rem;
        opacity: 0.95;
        font-weight: 300;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.75rem 1.25rem;
        border-radius: 30px;
        font-weight: 500;
        margin: 0.25rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        animation: fadeIn 0.5s ease-out;
        backdrop-filter: blur(10px);
    }
    
    .status-indicator:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .status-success {
        background: linear-gradient(135deg, rgba(220, 252, 231, 0.9) 0%, rgba(187, 247, 208, 0.9) 100%);
        color: #166534;
        border: 1px solid rgba(187, 247, 208, 0.5);
    }
    
    .status-warning {
        background: linear-gradient(135deg, rgba(254, 243, 199, 0.9) 0%, rgba(253, 230, 138, 0.9) 100%);
        color: #92400e;
        border: 1px solid rgba(253, 230, 138, 0.5);
    }
    
    .status-error {
        background: linear-gradient(135deg, rgba(254, 226, 226, 0.9) 0%, rgba(254, 202, 202, 0.9) 100%);
        color: #991b1b;
        border: 1px solid rgba(254, 202, 202, 0.5);
    }
    
    .user-message {
        background: linear-gradient(135deg, #ffffff 0%, #e8f2ff 100%);
        padding: 1.25rem;
        border-radius: 16px;
        border-left: 4px solid #3b82f6;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .user-message:hover {
        transform: translateX(4px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.15);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.25rem;
        border-radius: 16px;
        border-left: 4px solid #059669;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .assistant-message:hover {
        transform: translateX(4px);
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.15);
    }
    
    .medical-disclaimer {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid rgba(254, 202, 202, 0.5);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 2rem 0;
        color: #991b1b;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.1);
        transition: all 0.3s ease;
    }
    
    .medical-disclaimer:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(239, 68, 68, 0.15);
    }
    
    .api-config {
        background: linear-gradient(135deg, rgba(239, 246, 255, 0.95) 0%, rgba(219, 234, 254, 0.95) 100%);
        border: 2px solid rgba(191, 219, 254, 0.5);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.1);
    }
    
    .knowledge-source {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .knowledge-source:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
        border-color: rgba(191, 219, 254, 0.8);
    }
    
    /* Streamlit button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 30px;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(249, 250, 251, 0.95) 0%, rgba(243, 244, 246, 0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        width: 100%;
        margin: 0.25rem 0;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid rgba(191, 219, 254, 0.3);
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.9);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Select box styling */
    .stSelectbox > div > div > div {
        border-radius: 12px;
        border: 2px solid rgba(191, 219, 254, 0.3);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: #667eea;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(249, 250, 251, 0.9) 100%);
        border: 1px solid rgba(229, 231, 235, 0.5);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(249, 250, 251, 0.9) 0%, rgba(243, 244, 246, 0.9) 100%);
        border-radius: 12px;
        border: 1px solid rgba(229, 231, 235, 0.5);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(243, 244, 246, 0.9) 0%, rgba(237, 238, 240, 0.9) 100%);
        border-color: rgba(191, 219, 254, 0.5);
    }
    
    /* Chat input styling */
    .stChatInput > div {
        border-radius: 16px;
        border: 2px solid rgba(191, 219, 254, 0.3);
        background: rgba(255, 255, 255, 0.95);
        transition: all 0.3s ease;
    }
    
    .stChatInput > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Loading spinner enhancement */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Pulse animation for live indicators */
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Header with enhanced animations
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
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown('<div class="status-indicator" style="background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); color: #4338ca; width: 100%; justify-content: center; font-size: 1.1rem; padding: 1rem; margin-bottom: 1rem;">📄 Loading comprehensive medical knowledge...</div>', unsafe_allow_html=True)
            progress_bar.progress(20)
        
        # Load medical documents
        medical_docs_text = load_comprehensive_medical_knowledge()
        
        # Convert to Document objects
        documents = [Document(page_content=doc, metadata={"source": f"medical_knowledge_{i}"}) 
                    for i, doc in enumerate(medical_docs_text)]
        
        progress_bar.progress(40)
        status_text.markdown('<div class="status-indicator" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); color: #78350f; width: 100%; justify-content: center; font-size: 1.1rem; padding: 1rem; margin-bottom: 1rem;">✂️ Processing medical content...</div>', unsafe_allow_html=True)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        
        document_chunks = text_splitter.split_documents(documents)
        
        progress_bar.progress(60)
        status_text.markdown('<div class="status-indicator" style="background: linear-gradient(135deg, #ddd6fe 0%, #c4b5fd 100%); color: #6b21a8; width: 100%; justify-content: center; font-size: 1.1rem; padding: 1rem; margin-bottom: 1rem;">🧮 Creating medical embeddings...</div>', unsafe_allow_html=True)
        
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
        status_text.markdown('<div class="status-indicator" style="background: linear-gradient(135deg, #fecaca 0%, #fbbf24 100%); color: #92400e; width: 100%; justify-content: center; font-size: 1.1rem; padding: 1rem; margin-bottom: 1rem;">🗄️ Building FAISS vector store...</div>', unsafe_allow_html=True)
        
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
        status_text.markdown('<div class="status-indicator status-success pulse" style="width: 100%; justify-content: center; font-size: 1.2rem; padding: 1rem; margin-bottom: 1rem;">✅ Medical knowledge base ready!</div>', unsafe_allow_html=True)
        
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
    # API Configuration in sidebar with enhanced styling
    with st.sidebar:
        st.markdown('<div class="api-config" style="animation: fadeIn 0.8s ease-out;">', unsafe_allow_html=True)
        st.markdown("### 🔑 AI Configuration")
        
        # Check for API key in environment first
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                help="Get your API key from https://platform.openai.com/api-keys",
                placeholder="sk-..."
            )
        else:
            st.markdown('<div class="status-indicator status-success" style="width: 100%; justify-content: center;">✅ API key loaded from environment</div>', unsafe_allow_html=True)
        
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
            st.markdown(f'<div class="status-indicator" style="background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); color: #4338ca; border: 1px solid #a5b4fc; width: 100%; justify-content: center; margin-top: 1rem;">🤖 Model: {model}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-indicator" style="background: linear-gradient(135deg, #ddd6fe 0%, #c4b5fd 100%); color: #6b21a8; border: 1px solid #a78bfa; width: 100%; justify-content: center;">📊 Embeddings: {embedding_info}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-warning" style="width: 100%; justify-content: center;">⚠️ Please add your OpenAI API key</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        max_tokens = st.slider("Response Length", 256, 1024, 512)
    
    # Initialize AI client
    ai_client = MedicalAIClient(api_key, use_openai_embeddings)
    
    # System status with enhanced cards
    st.markdown('<div style="margin: 1.5rem 0;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ai_client.is_configured():
            st.markdown('<div class="status-indicator status-success pulse" style="width: 100%; justify-content: center;">🟢 AI Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-warning" style="width: 100%; justify-content: center;">🟡 API Key Needed</div>', unsafe_allow_html=True)
    
    with col2:
        session_duration = datetime.now() - st.session_state.session_start
        st.markdown(f'<div class="status-indicator" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); color: #78350f; border: 1px solid #fcd34d; width: 100%; justify-content: center;">⏱️ {str(session_duration).split(".")[0]}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="status-indicator" style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); color: #1e3a8a; border: 1px solid #93c5fd; width: 100%; justify-content: center;">💬 Queries: {st.session_state.query_count}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize knowledge base with enhanced loading experience
    if not st.session_state.system_initialized:
        if not api_key and use_openai_embeddings:
            st.markdown('<div class="status-indicator status-warning" style="width: 100%; justify-content: center; margin: 1rem 0;">⚠️ OpenAI API key required for OpenAI embeddings. Using SentenceTransformers instead.</div>', unsafe_allow_html=True)
            use_openai_embeddings = False
        
        with st.spinner("🚀 Initializing medical knowledge base with FAISS..."):
            retriever = setup_medical_vectorstore(use_openai_embeddings, api_key)
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.system_initialized = True
                st.markdown('<div class="status-indicator status-success pulse" style="width: 100%; justify-content: center; margin: 1rem 0; font-size: 1.1rem; padding: 1rem 2rem;">✅ MedAssist AI is ready for medical consultations!</div>', unsafe_allow_html=True)
                time.sleep(1)
                st.rerun()
    
    # Sidebar info with enhanced styling
    with st.sidebar:
        if hasattr(st.session_state, 'total_chunks'):
            st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
            st.markdown("### 📊 Knowledge Base")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Topics", st.session_state.total_topics)
            with col_b:
                st.metric("Chunks", st.session_state.total_chunks)
            
            if hasattr(st.session_state, 'embedding_model'):
                st.markdown(f'<div class="status-indicator" style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); color: #6b21a8; border: 1px solid #d8b4fe; width: 100%; justify-content: center; margin-top: 1rem;">🔬 Using: {st.session_state.embedding_model}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
        st.markdown("### 🩺 Sample Medical Questions")
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat interface with enhanced styling
    st.markdown('<h2 style="color: #4338ca; font-weight: 600; margin: 2rem 0 1.5rem 0;">💬 Medical Consultation</h2>', unsafe_allow_html=True)
    
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
            st.markdown('<div class="status-indicator status-error" style="width: 100%; justify-content: center; margin: 1rem 0;">Please configure your OpenAI API key first!</div>', unsafe_allow_html=True)
            return
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.query_count += 1
        
        with st.spinner("🔍 Consulting medical knowledge base with FAISS..."):
            # Add a subtle loading animation
            loading_placeholder = st.empty()
            loading_placeholder.markdown('<div class="status-indicator pulse" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 100%; justify-content: center; margin: 1rem 0;">🤔 Analyzing your medical query...</div>', unsafe_allow_html=True)
            
            response = generate_medical_rag_response(
                ai_client, 
                st.session_state.retriever, 
                user_input, 
                model=model, 
                max_tokens=max_tokens
            )
            
            loading_placeholder.empty()
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Show knowledge sources if response exists with enhanced styling
    if st.session_state.messages and st.session_state.system_initialized:
        with st.expander("🔍 View Knowledge Sources (Last Query)"):
            if st.session_state.messages:
                last_question = [msg for msg in st.session_state.messages if msg["role"] == "user"]
                if last_question:
                    docs = st.session_state.retriever.get_relevant_documents(last_question[-1]["content"])
                    for i, doc in enumerate(docs):
                        st.markdown(f"""
                        <div class="knowledge-source" style="animation: fadeIn 0.5s ease-out {0.1 * i}s both;">
                            <strong style="color: #6366f1;">Source {i+1}:</strong><br>
                            <span style="color: #475569; line-height: 1.6;">{doc.page_content[:400]}...</span>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Medical disclaimer with enhanced styling
    st.markdown("""
    <div class="medical-disclaimer" style="animation: fadeIn 1s ease-out 0.5s both;">
        <strong style="font-size: 1.1rem;">⚠️ Important Medical Disclaimer</strong><br>
        <span style="line-height: 1.6; opacity: 0.9;">This AI assistant provides information for educational purposes only and should not replace professional medical advice, diagnosis, or treatment. 
        The responses are based on medical knowledge and should be verified against current medical literature and clinical guidelines.
        Always consult with qualified healthcare professionals for medical decisions. In case of medical emergencies, contact emergency services immediately.</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()