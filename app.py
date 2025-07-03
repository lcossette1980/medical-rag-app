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

# Initialize theme state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Dark mode toggle in the top right
col_spacer, col_theme = st.columns([10, 1])
with col_theme:
    if st.button('🌓', help='Toggle dark mode', key='theme_toggle'):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Enhanced CSS with dynamic theme support
if st.session_state.dark_mode:
    bg_gradient = 'linear-gradient(135deg, var(--dark-bg) 0%, var(--dark-surface) 100%)'
    text_color = 'var(--dark-text)'
    header_bg = 'linear-gradient(135deg, var(--dark-surface) 0%, #3a3a3a 100%)'
    header_shadow = '0 24px 48px var(--dark-shadow)'
    header_border = 'var(--dark-border)'
    card_bg = 'var(--dark-surface)'
    card_border = 'var(--dark-border)'
    card_shadow = '0 8px 24px var(--dark-shadow)'
    user_msg_bg = 'linear-gradient(135deg, #3a3a3a 0%, #2d2d2d 100%)'
    assistant_msg_bg = 'linear-gradient(135deg, var(--dark-surface) 0%, rgba(164, 74, 63, 0.1) 100%)'
    sidebar_bg = 'linear-gradient(135deg, var(--dark-surface) 0%, #3a3a3a 100%)'
else:
    bg_gradient = 'linear-gradient(135deg, var(--bone) 0%, var(--pearl) 100%)'
    text_color = 'var(--charcoal)'
    header_bg = 'linear-gradient(135deg, var(--charcoal) 0%, #404040 100%)'
    header_shadow = '0 24px 48px var(--shadow-heavy)'
    header_border = 'rgba(164, 74, 63, 0.2)'
    card_bg = 'var(--white)'
    card_border = 'var(--pearl)'
    card_shadow = '0 8px 24px var(--shadow-light)'
    user_msg_bg = 'linear-gradient(135deg, var(--bone) 0%, var(--pearl) 100%)'
    assistant_msg_bg = 'linear-gradient(135deg, var(--white) 0%, rgba(164, 74, 63, 0.03) 100%)'
    sidebar_bg = 'linear-gradient(135deg, var(--white) 0%, var(--bone) 100%)'

# Enhanced CSS with dynamic theme support
# Build CSS with theme variables
theme_css = f"""
    .stApp {{
        background: {bg_gradient};
        min-height: 100vh;
        color: {text_color};
    }}
    
    .main-header {{
        background: {header_bg};
        box-shadow: {header_shadow};
        border: 1px solid {header_border};
    }}
    
    .status-card {{
        background: {card_bg};
        border: 2px solid {card_border};
        box-shadow: {card_shadow};
    }}
    
    .chat-container {{
        background: {card_bg};
        box-shadow: {card_shadow};
        border: 1px solid {card_border};
    }}
    
    .user-message {{
        background: {user_msg_bg};
        box-shadow: {card_shadow};
    }}
    
    .assistant-message {{
        background: {assistant_msg_bg};
        box-shadow: {card_shadow};
    }}
    
    .message-content {{
        color: {text_color};
    }}
    
    .message-label {{
        color: {text_color};
    }}
    
    .sidebar-config {{
        background: {sidebar_bg};
        border: 2px solid {card_border};
    }}
"""

st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)

# Static CSS without theme variables
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Lato:wght@300;400;500;600&display=swap');
    
    :root {{
        --charcoal: #2A2A2A;
        --chestnut: #A44A3F;
        --khaki: #A59E8C;
        --pearl: #D7CEB2;
        --bone: #F5F2EA;
        --white: #FFFFFF;
        --shadow-light: rgba(42, 42, 42, 0.08);
        --shadow-medium: rgba(42, 42, 42, 0.15);
        --shadow-heavy: rgba(42, 42, 42, 0.25);
        
        --dark-bg: #1a1a1a;
        --dark-surface: #2d2d2d;
        --dark-text: #e0e0e0;
        --dark-border: #3d3d3d;
        --dark-shadow: rgba(0, 0, 0, 0.3);
    }}
    
    * {{
        font-family: 'Lato', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    .stApp {{
        background: {bg_gradient};
        min-height: 100vh;
        color: {text_color};
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    .main-header {{
        background: {header_bg};
        padding: 3.5rem 2.5rem;
        border-radius: 24px;
        color: var(--bone);
        margin-bottom: 3rem;
        text-align: center;
        box-shadow: {header_shadow};
        position: relative;
        overflow: hidden;
        animation: slideInDown 1s ease-out;
        border: 1px solid {header_border};
    }}
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {{
        .main-header {{
            padding: 2rem 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .main-header h1 {{
            font-size: 2.5rem !important;
        }}
        
        .main-header .subtitle {{
            font-size: 1.1rem !important;
        }}
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--chestnut) 0%%, var(--khaki) 50%%, var(--chestnut) 100%%);
        animation: shimmer 3s ease-in-out infinite;
    }}
    
    @keyframes shimmer {{
        0%%, 100%% {{ opacity: 0.6; }}
        50%% {{ opacity: 1; }}
    }}
    
    @keyframes slideInDown {{
        from {{
            opacity: 0;
            transform: translateY(-40px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes pulseGlow {{
        0%%, 100%% {{ 
            box-shadow: 0 0 20px rgba(164, 74, 63, 0.3);
        }}
        50%% {{ 
            box-shadow: 0 0 30px rgba(164, 74, 63, 0.5);
        }}
    }}
    
    .main-header h1 {{
        margin: 0;
        font-family: 'Playfair Display', serif;
        font-size: 3.8rem;
        font-weight: 900;
        text-shadow: 2px 4px 8px rgba(0,0,0,0.3);
        letter-spacing: -2px;
        background: linear-gradient(135deg, var(--bone) 0%%, var(--pearl) 100%%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .main-header .subtitle {{
        margin: 1.5rem 0 0 0;
        font-family: 'Lato', sans-serif;
        font-size: 1.4rem;
        opacity: 0.9;
        font-weight: 400;
        letter-spacing: 1px;
        color: var(--pearl);
    }}
    
    .medical-badge {{
        display: inline-flex;
        align-items: center;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        filter: drop-shadow(2px 4px 8px rgba(0,0,0,0.2));
    }}
    
    .status-card {{
        background: {card_bg};
        border: 2px solid {card_border};
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: {card_shadow};
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out;
    }}
    
    /* Mobile card adjustments */
    @media (max-width: 768px) {{
        .status-card {{
            padding: 1rem;
            margin: 0.5rem 0;
        }}
    }}
    
    .status-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%%;
        height: 3px;
        background: linear-gradient(90deg, var(--chestnut) 0%%, var(--khaki) 100%%);
    }}
    
    .status-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 16px 40px var(--shadow-medium);
        border-color: var(--chestnut);
    }}
    
    .status-success {{
        border-left: 6px solid var(--chestnut);
        background: linear-gradient(135deg, var(--white) 0%%, rgba(164, 74, 63, 0.02) 100%%);
    }}
    
    .status-warning {{
        border-left: 6px solid var(--khaki);
        background: linear-gradient(135deg, var(--white) 0%%, rgba(165, 158, 140, 0.02) 100%%);
    }}
    
    .status-error {{
        border-left: 6px solid #d32f2f;
        background: linear-gradient(135deg, var(--white) 0%%, rgba(211, 47, 47, 0.02) 100%%);
    }}
    
    .status-text {
        font-family: 'Lato', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        color: var(--charcoal);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .status-icon {
        font-size: 1.25rem;
        filter: drop-shadow(1px 2px 4px rgba(0,0,0,0.1));
    }
    
    .chat-container {{
        background: {card_bg};
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: {card_shadow};
        border: 1px solid {card_border};
        position: relative;
    }}
    
    @media (max-width: 768px) {{
        .chat-container {{
            padding: 1rem;
            border-radius: 15px;
        }}
    }}
    
    .chat-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--chestnut) 0%%, var(--khaki) 50%%, var(--chestnut) 100%%);
        border-radius: 20px 20px 0 0;
    }
    
    .user-message {{
        background: {user_msg_bg};
        border: 2px solid var(--khaki);
        padding: 1.75rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: {card_shadow};
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
        position: relative;
    }}
    
    .user-message::before {
        content: '👩‍⚕️';
        position: absolute;
        top: -12px;
        left: 20px;
        background: var(--white);
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 1.2rem;
        box-shadow: 0 4px 12px var(--shadow-light);
    }
    
    .user-message:hover {
        transform: translateX(8px);
        box-shadow: 0 12px 32px var(--shadow-medium);
    }
    
    .assistant-message {{
        background: {assistant_msg_bg};
        border: 2px solid var(--chestnut);
        padding: 1.75rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: {card_shadow};
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
        position: relative;
    }}
    
    .assistant-message::before {
        content: '🩺';
        position: absolute;
        top: -12px;
        left: 20px;
        background: var(--chestnut);
        color: var(--white);
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 1.2rem;
        box-shadow: 0 4px 12px var(--shadow-medium);
    }
    
    .assistant-message:hover {
        transform: translateX(8px);
        box-shadow: 0 12px 32px var(--shadow-medium);
    }
    
    .message-content {{
        font-family: 'Lato', sans-serif;
        line-height: 1.7;
        color: {text_color};
        margin-top: 0.5rem;
    }}
    
    .message-label {{
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: {text_color};
        margin-bottom: 0.5rem;
    }}
    
    .medical-disclaimer {
        background: linear-gradient(135deg, #fef7f7 0%, #fef0f0 100%);
        border: 2px solid #ffcdd2;
        border-radius: 20px;
        padding: 2rem;
        margin: 3rem 0;
        color: #b71c1c;
        box-shadow: 0 12px 32px rgba(183, 28, 28, 0.1);
        position: relative;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .medical-disclaimer::before {
        content: '⚠️';
        position: absolute;
        top: -15px;
        left: 30px;
        background: #ffebee;
        padding: 10px 15px;
        border-radius: 25px;
        font-size: 1.5rem;
        box-shadow: 0 4px 12px rgba(183, 28, 28, 0.2);
        border: 2px solid #ffcdd2;
    }
    
    .disclaimer-title {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        color: #b71c1c;
    }
    
    .disclaimer-text {
        font-family: 'Lato', sans-serif;
        line-height: 1.6;
        opacity: 0.9;
    }
    
    .sidebar-config {{
        background: {sidebar_bg};
        border: 2px solid {card_border};
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px var(--shadow-light);
        position: relative;
    }
    
    .sidebar-config::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--chestnut) 0%%, var(--khaki) 100%%);
        border-radius: 20px 20px 0 0;
    }
    
    .config-title {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 1.4rem;
        color: var(--charcoal);
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .knowledge-source {
        background: var(--white);
        border: 2px solid var(--pearl);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px var(--shadow-light);
        position: relative;
    }
    
    .knowledge-source::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--chestnut) 0%, var(--khaki) 100%);
        border-radius: 16px 0 0 16px;
    }
    
    .knowledge-source:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px var(--shadow-medium);
        border-color: var(--chestnut);
    }
    
    .source-label {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: var(--chestnut);
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }
    
    .source-content {
        font-family: 'Lato', sans-serif;
        color: var(--charcoal);
        line-height: 1.6;
        opacity: 0.9;
    }
    
    /* Enhanced Streamlit component styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--chestnut) 0%, #b85550 100%);
        color: var(--white);
        border: none;
        padding: 0.9rem 2rem;
        border-radius: 25px;
        font-family: 'Lato', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 16px rgba(164, 74, 63, 0.3);
        border: 2px solid transparent;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(164, 74, 63, 0.4);
        background: linear-gradient(135deg, #b85550 0%, var(--chestnut) 100%);
        border-color: var(--khaki);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bone) 0%, var(--pearl) 100%);
        border-right: 3px solid var(--khaki);
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, var(--khaki) 0%, #b5ae9b 100%);
        width: 100%%;
        margin: 0.4rem 0;
        font-size: 0.9rem;
        padding: 0.8rem 1.5rem;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #b5ae9b 0%, var(--khaki) 100%);
        color: var(--charcoal);
    }
    
    /* Enhanced input styling */
    .stTextInput > div > div > input {
        border-radius: 16px;
        border: 2px solid var(--pearl);
        padding: 1rem 1.5rem;
        transition: all 0.3s ease;
        background: var(--white);
        font-family: 'Lato', sans-serif;
        color: var(--charcoal);
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--chestnut);
        box-shadow: 0 0 0 4px rgba(164, 74, 63, 0.1);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--khaki);
        opacity: 0.7;
    }
    
    /* Select box styling */
    .stSelectbox > div > div > div {
        border-radius: 16px;
        border: 2px solid var(--pearl);
        transition: all 0.3s ease;
        background: var(--white);
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: var(--chestnut);
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--white) 0%, var(--bone) 100%);
        border: 2px solid var(--pearl);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px var(--shadow-light);
        transition: all 0.3s ease;
        position: relative;
    }
    
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%%;
        height: 3px;
        background: linear-gradient(90deg, var(--chestnut) 0%%, var(--khaki) 100%%);
        border-radius: 16px 16px 0 0;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px var(--shadow-medium);
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: var(--chestnut);
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        font-family: 'Lato', sans-serif;
        color: var(--charcoal);
        font-weight: 500;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--chestnut) 0%%, var(--khaki) 100%%);
        border-radius: 10px;
    }
    
    .stProgress > div > div > div {
        background-color: var(--pearl);
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, var(--white) 0%, var(--bone) 100%);
        border-radius: 16px;
        border: 2px solid var(--pearl);
        transition: all 0.3s ease;
        padding: 1rem 1.5rem;
        font-family: 'Lato', sans-serif;
        font-weight: 600;
        color: var(--charcoal);
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, var(--bone) 0%, var(--pearl) 100%);
        border-color: var(--chestnut);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px var(--shadow-light);
    }
    
    /* Chat input styling */
    .stChatInput > div {
        border-radius: 20px;
        border: 3px solid var(--pearl);
        background: var(--white);
        transition: all 0.3s ease;
        padding: 0.5rem;
    }
    
    .stChatInput > div:focus-within {
        border-color: var(--chestnut);
        box-shadow: 0 0 0 4px rgba(164, 74, 63, 0.1);
    }
    
    .stChatInput input {
        font-family: 'Lato', sans-serif;
        font-size: 1.1rem;
        color: var(--charcoal);
        padding: 1rem 1.5rem;
    }
    
    .stChatInput input::placeholder {
        color: var(--khaki);
        opacity: 0.8;
    }
    
    /* Loading spinner enhancement */
    .stSpinner > div {
        border-color: var(--chestnut) !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--pearl);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--chestnut) 0%, var(--khaki) 100%);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--khaki) 0%, var(--chestnut) 100%);
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 2rem;
        color: var(--charcoal);
        margin: 2rem 0 1.5rem 0;
        text-align: center;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, var(--chestnut) 0%%, var(--khaki) 100%%);
        border-radius: 3px;
    }
    
    /* Sample questions styling */
    .sample-questions {
        background: linear-gradient(135deg, var(--white) 0%, var(--bone) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid var(--pearl);
        box-shadow: 0 8px 24px var(--shadow-light);
    }
    
    .sample-questions h3 {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: var(--charcoal);
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Loading states */
    .loading-indicator {
        background: linear-gradient(135deg, var(--chestnut) 0%, var(--khaki) 100%);
        color: var(--white);
        padding: 1rem 2rem;
        border-radius: 25px;
        font-family: 'Lato', sans-serif;
        font-weight: 500;
        text-align: center;
        margin: 1rem 0;
        animation: pulseGlow 2s infinite;
        box-shadow: 0 8px 24px rgba(164, 74, 63, 0.3);
    }
    
    /* Typography for markdown content */
    .markdown-content h1 {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: var(--charcoal);
    }
    
    .markdown-content h2 {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: var(--chestnut);
        border-bottom: 2px solid var(--pearl);
        padding-bottom: 0.5rem;
    }
    
    .markdown-content h3 {
        font-family: 'Lato', sans-serif;
        font-weight: 600;
        color: var(--charcoal);
    }
    
    .markdown-content p {
        font-family: 'Lato', sans-serif;
        line-height: 1.7;
        color: var(--charcoal);
    }
    
    .markdown-content strong {
        color: var(--chestnut);
        font-weight: 600;
    }
    
    .markdown-content em {
        color: var(--khaki);
        font-style: italic;
    }
    
    .markdown-content ul, .markdown-content ol {
        font-family: 'Lato', sans-serif;
        color: var(--charcoal);
        line-height: 1.6;
    }
    
    .markdown-content li {
        margin-bottom: 0.5rem;
    }
    
    .markdown-content code {
        background: var(--pearl);
        color: var(--charcoal);
        padding: 0.2rem 0.4rem;
        border-radius: 6px;
        font-family: 'Monaco', 'Consolas', monospace;
    }
    
    /* Print-friendly styles */
    @media print {{
        /* Hide unnecessary elements */
        .stSidebar, button, .stButton, [data-testid="stToolbar"], 
        [data-testid="stStatusWidget"], .medical-disclaimer,
        .status-card, details, summary {{
            display: none !important;
        }}
        
        /* Optimize layout for printing */
        .stApp {{
            background: white !important;
            color: black !important;
        }}
        
        .main-header {{
            background: white !important;
            color: black !important;
            box-shadow: none !important;
            border: 2px solid black !important;
            page-break-after: avoid;
        }}
        
        .main-header h1 {{
            color: black !important;
            -webkit-text-fill-color: black !important;
        }}
        
        .chat-container {{
            background: white !important;
            box-shadow: none !important;
            border: 1px solid #ccc !important;
        }}
        
        .user-message, .assistant-message {{
            background: white !important;
            border: 1px solid #999 !important;
            box-shadow: none !important;
            page-break-inside: avoid;
        }}
        
        .message-content {{
            color: black !important;
        }}
        
        .message-label {{
            color: black !important;
            font-weight: bold !important;
        }}
        
        /* Ensure proper page breaks */
        .user-message {{
            page-break-after: auto;
        }}
        
        .assistant-message {{
            page-break-after: always;
        }}
        
        /* Add print header */
        @page {{
            margin: 1in;
            @top-center {{
                content: "MedAssist AI Consultation - " attr(data-date);
            }}
        }}
    }}
    
    .markdown-content pre {{
        background: var(--bone);
        border: 2px solid var(--pearl);
        border-radius: 12px;
        padding: 1rem;
        overflow-x: auto;
    }}
</style>
""", unsafe_allow_html=True)

# Enhanced Header with your branding
st.markdown("""
<div class="main-header">
    <div class="medical-badge">🩺</div>
    <h1>MedAssist AI</h1>
    <p class="subtitle">Advanced Medical Question Answering System</p>
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
            
            status_text.markdown('<div class="loading-indicator">📄 Loading comprehensive medical knowledge...</div>', unsafe_allow_html=True)
            progress_bar.progress(20)
        
        # Load medical documents
        medical_docs_text = load_comprehensive_medical_knowledge()
        
        # Convert to Document objects
        documents = [Document(page_content=doc, metadata={"source": f"medical_knowledge_{i}"}) 
                    for i, doc in enumerate(medical_docs_text)]
        
        progress_bar.progress(40)
        status_text.markdown('<div class="loading-indicator">✂️ Processing medical content...</div>', unsafe_allow_html=True)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        
        document_chunks = text_splitter.split_documents(documents)
        
        progress_bar.progress(60)
        status_text.markdown('<div class="loading-indicator">🧮 Creating medical embeddings...</div>', unsafe_allow_html=True)
        
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
        status_text.markdown('<div class="loading-indicator">🗄️ Building FAISS vector store...</div>', unsafe_allow_html=True)
        
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
        status_text.markdown('<div class="status-card status-success"><div class="status-text"><span class="status-icon">✅</span>Medical knowledge base ready!</div></div>', unsafe_allow_html=True)
        
        # Store stats
        st.session_state.total_chunks = len(document_chunks)
        st.session_state.total_topics = len(medical_docs_text)
        st.session_state.embedding_model = "OpenAI" if use_openai_embeddings else "SentenceTransformer"
        
        time.sleep(1.5)
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
    # Show onboarding for first-time users
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
    
    if st.session_state.first_visit:
        with st.container():
            st.markdown("""
            <div class="onboarding-container" style="
                background: linear-gradient(135deg, var(--pearl) 0%, var(--bone) 100%);
                border: 2px solid var(--chestnut);
                border-radius: 20px;
                padding: 2rem;
                margin-bottom: 2rem;
                text-align: center;
            ">
                <h2 style="color: var(--chestnut); margin-bottom: 1rem;">🎆 Welcome to MedAssist AI!</h2>
                <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">
                    Your intelligent medical consultation assistant powered by advanced AI and a comprehensive medical knowledge base.
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <h4>🔒 Secure</h4>
                        <p>Your data stays private</p>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <h4>🎯 Accurate</h4>
                        <p>Evidence-based responses</p>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <h4>⚡ Fast</h4>
                        <p>Instant medical insights</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button('🚀 Get Started', use_container_width=True):
                st.session_state.first_visit = False
                st.rerun()
    
    # API Configuration in sidebar with enhanced styling
    with st.sidebar:
        st.markdown('<div class="sidebar-config">', unsafe_allow_html=True)
        st.markdown('<div class="config-title">🔑 AI Configuration</div>', unsafe_allow_html=True)
        
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
            st.markdown('<div class="status-card status-success"><div class="status-text"><span class="status-icon">✅</span>API key loaded from environment</div></div>', unsafe_allow_html=True)
        
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
            st.markdown(f'<div class="status-card"><div class="status-text"><span class="status-icon">🤖</span>Model: {model}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-card"><div class="status-text"><span class="status-icon">📊</span>Embeddings: {embedding_info}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card status-warning"><div class="status-text"><span class="status-icon">⚠️</span>Please add your OpenAI API key</div></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        max_tokens = st.slider("Response Length", 256, 1024, 512)
    
    # Initialize AI client
    ai_client = MedicalAIClient(api_key, use_openai_embeddings)
    
    # System status with enhanced cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ai_client.is_configured():
            st.markdown('<div class="status-card status-success"><div class="status-text"><span class="status-icon">🟢</span>AI Connected</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card status-warning"><div class="status-text"><span class="status-icon">🟡</span>API Key Needed</div></div>', unsafe_allow_html=True)
    
    with col2:
        session_duration = datetime.now() - st.session_state.session_start
        st.markdown(f'<div class="status-card"><div class="status-text"><span class="status-icon">⏱️</span>{str(session_duration).split(".")[0]}</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="status-card"><div class="status-text"><span class="status-icon">💬</span>Queries: {st.session_state.query_count}</div></div>', unsafe_allow_html=True)
    
    # Initialize knowledge base with enhanced loading experience
    if not st.session_state.system_initialized:
        if not api_key and use_openai_embeddings:
            st.markdown('<div class="status-card status-warning"><div class="status-text"><span class="status-icon">⚠️</span>OpenAI API key required for OpenAI embeddings. Using SentenceTransformers instead.</div></div>', unsafe_allow_html=True)
            use_openai_embeddings = False
        
        with st.spinner("🚀 Initializing medical knowledge base with FAISS..."):
            retriever = setup_medical_vectorstore(use_openai_embeddings, api_key)
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.system_initialized = True
                st.markdown('<div class="status-card status-success"><div class="status-text"><span class="status-icon">✅</span>MedAssist AI is ready for medical consultations!</div></div>', unsafe_allow_html=True)
                time.sleep(1)
                st.rerun()
    
    # Sidebar info with enhanced styling
    with st.sidebar:
        if hasattr(st.session_state, 'total_chunks'):
            st.markdown('<div class="sidebar-config" style="margin-top: 2rem;">', unsafe_allow_html=True)
            st.markdown('<div class="config-title">📊 Knowledge Base</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Topics", st.session_state.total_topics)
            with col_b:
                st.metric("Chunks", st.session_state.total_chunks)
            
            if hasattr(st.session_state, 'embedding_model'):
                st.markdown(f'<div class="status-card"><div class="status-text"><span class="status-icon">🔬</span>Using: {st.session_state.embedding_model}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sample-questions">', unsafe_allow_html=True)
        st.markdown('<h3>🩺 Sample Medical Questions</h3>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-header">💬 Medical Consultation</div>', unsafe_allow_html=True)
    
    # Chat export functionality
    if st.session_state.messages:
        col1, col2, col3 = st.columns([4, 1, 1])
        with col2:
            if st.button('📥 Export Chat', help='Export conversation as markdown'):
                # Generate markdown export
                export_text = "# MedAssist AI Consultation\n\n"
                export_text += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                for msg in st.session_state.messages:
                    role = "Healthcare Professional" if msg["role"] == "user" else "MedAssist AI"
                    export_text += f"## {role}\n{msg['content']}\n\n---\n\n"
                
                # Add disclaimer
                export_text += "\n\n> **Disclaimer:** This AI assistant provides information for educational purposes only and should not replace professional medical advice."
                
                # Download button
                st.download_button(
                    label="💾 Download",
                    data=export_text,
                    file_name=f"medical_consultation_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )
        
        with col3:
            if st.button('🗑️ Clear Chat', help='Clear conversation history'):
                st.session_state.messages = []
                st.session_state.query_count = 0
                st.rerun()
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="message-label">Healthcare Professional</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <div class="message-label">MedAssist AI</div>
                <div class="message-content markdown-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("💬 Ask a medical question...")
    
    # Handle sample question selection
    if hasattr(st.session_state, 'user_input') and st.session_state.user_input:
        user_input = st.session_state.user_input
        del st.session_state.user_input
    
    if user_input and st.session_state.system_initialized:
        if not ai_client.is_configured():
            st.markdown('<div class="status-card status-error"><div class="status-text"><span class="status-icon">❌</span>Please configure your OpenAI API key first!</div></div>', unsafe_allow_html=True)
            return
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.query_count += 1
        
        with st.spinner("🔍 Consulting medical knowledge base with FAISS..."):
            # Add a subtle loading animation
            loading_placeholder = st.empty()
            loading_placeholder.markdown('<div class="loading-indicator">🤔 Analyzing your medical query...</div>', unsafe_allow_html=True)
            
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
        with st.expander("🔍 View Knowledge Sources & Citations"):
            if st.session_state.messages:
                last_question = [msg for msg in st.session_state.messages if msg["role"] == "user"]
                if last_question:
                    docs = st.session_state.retriever.get_relevant_documents(last_question[-1]["content"])
                    
                    # Add relevance scores and better formatting
                    st.markdown("<h4>📚 Most Relevant Medical Sources</h4>", unsafe_allow_html=True)
                    
                    for i, doc in enumerate(docs[:5]):  # Limit to top 5 sources
                        # Calculate simple relevance score (you can improve this)
                        relevance = f"{(5-i)*20}%"
                        
                        st.markdown(f"""
                        <div class="knowledge-source" style="animation: fadeInUp 0.5s ease-out {0.1 * i}s both;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <div class="source-label">📖 Source {i+1}</div>
                                <div style="color: var(--chestnut); font-weight: bold;">Relevance: {relevance}</div>
                            </div>
                            <div class="source-content">
                                <div style="margin-bottom: 0.5rem; font-style: italic; color: #666;">
                                    {doc.metadata.get('source', 'Medical Knowledge Base')}
                                </div>
                                {doc.page_content[:500]}...
                            </div>
                            <details style="margin-top: 1rem;">
                                <summary style="cursor: pointer; color: var(--chestnut); font-weight: 500;">View Full Context</summary>
                                <div style="padding: 1rem; background: rgba(0,0,0,0.02); border-radius: 8px; margin-top: 0.5rem;">
                                    {doc.page_content}
                                </div>
                            </details>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Medical disclaimer with enhanced styling
    st.markdown("""
    <div class="medical-disclaimer">
        <div class="disclaimer-title">Important Medical Disclaimer</div>
        <div class="disclaimer-text">This AI assistant provides information for educational purposes only and should not replace professional medical advice, diagnosis, or treatment. 
        The responses are based on medical knowledge and should be verified against current medical literature and clinical guidelines.
        Always consult with qualified healthcare professionals for medical decisions. In case of medical emergencies, contact emergency services immediately.</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()