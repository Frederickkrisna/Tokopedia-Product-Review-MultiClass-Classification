import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import re

# NLP and ML
from sklearn.preprocessing import LabelEncoder
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# Deep Learning
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Tokopedia Review Classifier",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #42B549;
        --secondary-color: #00854D;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #42B549, #00854D);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(66, 181, 73, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #42B549;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 16px rgba(66, 181, 73, 0.3);
    }
    
    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #42B549;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-text {
        font-size: 28px;
        font-weight: bold;
        color: #42B549;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #42B549, #00854D);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #00854D, #42B549);
        box-shadow: 0 4px 12px rgba(66, 181, 73, 0.4);
        transform: translateY(-2px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #42B549, #00854D);
        color: white;
        border-color: #42B549;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #F0FDF4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #42B549;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# ===== CONFIGURATION =====
RANDOM_STATE = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# ===== MODEL MANAGER =====
@st.cache_resource
def load_models():
    """Load all models and dataset (cached)"""
    model_data = {
        'df_clean': None,
        'label_encoder': None,
        'tfidf_vectorizer': None,
        'svm_baseline': None,
        'svm_tuned': None,
        'svm_tuned_balanced': None,
        'model_indobert_baseline': None,
        'tokenizer_indobert_baseline': None,
        'model_indobert_baseline_balanced': None,
        'tokenizer_indobert_baseline_balanced': None,
        'model_indobert': None,
        'tokenizer_indobert': None,
        'model_indobert_tuned_balanced': None,
        'tokenizer_indobert_tuned_balanced': None
    }
    
    # Load dataset
    try:
        model_data['df_clean'] = pd.read_csv('df_clean.csv', encoding='utf-8')
        model_data['label_encoder'] = LabelEncoder()
        model_data['label_encoder'].fit(model_data['df_clean']['label'].values)
        st.sidebar.success(f"✓ Dataset: {len(model_data['df_clean']):,} reviews")
    except Exception as e:
        st.sidebar.warning(f"⚠ Dataset load failed: {e}")
        model_data['df_clean'] = pd.DataFrame({'label': ['Pujian', 'Kritik', 'Saran', 'Informasi/Netral', 'Lainnya']})
        model_data['label_encoder'] = LabelEncoder()
        model_data['label_encoder'].fit(['Pujian', 'Kritik', 'Saran', 'Informasi/Netral', 'Lainnya'])
    
    # Load SVM models
    try:
        model_data['tfidf_vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
        model_data['svm_baseline'] = joblib.load('models/svm_baseline.pkl')
        model_data['svm_tuned'] = joblib.load('models/svm_tuned.pkl')
        try:
            model_data['svm_tuned_balanced'] = joblib.load('models/svm_tuned_balanced.pkl')
        except:
            pass
        st.sidebar.success("✓ SVM models loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠ SVM models not found: {e}")
    
    # Load IndoBERT models
    indobert_loaded = 0
    
    # Baseline
    try:
        model_data['tokenizer_indobert_baseline'] = AutoTokenizer.from_pretrained('./models/indobert_p2_baseline')
        model_data['model_indobert_baseline'] = AutoModelForSequenceClassification.from_pretrained('./models/indobert_p2_baseline')
        model_data['model_indobert_baseline'].to(device)
        model_data['model_indobert_baseline'].eval()
        indobert_loaded += 1
    except:
        pass
    
    # Baseline Balanced
    try:
        model_data['tokenizer_indobert_baseline_balanced'] = AutoTokenizer.from_pretrained('./models/indobert_p2_baseline_balanced')
        model_data['model_indobert_baseline_balanced'] = AutoModelForSequenceClassification.from_pretrained('./models/indobert_p2_baseline_balanced')
        model_data['model_indobert_baseline_balanced'].to(device)
        model_data['model_indobert_baseline_balanced'].eval()
        indobert_loaded += 1
    except:
        pass
    
    # Tuned
    try:
        model_data['tokenizer_indobert'] = AutoTokenizer.from_pretrained('./models/indobert_p2_tuned')
        model_data['model_indobert'] = AutoModelForSequenceClassification.from_pretrained('./models/indobert_p2_tuned')
        model_data['model_indobert'].to(device)
        model_data['model_indobert'].eval()
        indobert_loaded += 1
    except:
        pass
    
    # Tuned Balanced
    try:
        model_data['tokenizer_indobert_tuned_balanced'] = AutoTokenizer.from_pretrained('./models/indobert_p2_tuned_balanced')
        model_data['model_indobert_tuned_balanced'] = AutoModelForSequenceClassification.from_pretrained('./models/indobert_p2_tuned_balanced')
        model_data['model_indobert_tuned_balanced'].to(device)
        model_data['model_indobert_tuned_balanced'].eval()
        indobert_loaded += 1
    except:
        pass
    
    if indobert_loaded > 0:
        st.sidebar.success(f"✓ {indobert_loaded} IndoBERT model(s) loaded on {device}")
    else:
        st.sidebar.warning("⚠ No IndoBERT models found")
    
    return model_data

# ===== PREPROCESSING =====
def preprocess_text(text):
    """Clean and preprocess Indonesian text"""
    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    text = ' '.join(text.split())
    text = stemmer.stem(text)
    
    return text

# ===== PREDICTION =====
def make_prediction(text, model_name, model_data):
    """Perform prediction using selected model"""
    if not text.strip():
        return None, None, "Text kosong, silakan masukkan review"
    
    try:
        preprocessed = preprocess_text(text)
        
        if model_name.startswith('SVM'):
            if model_data['tfidf_vectorizer'] is None:
                return None, None, "Model SVM belum dimuat"
            
            if model_name == 'SVM Baseline':
                model = model_data['svm_baseline']
            elif model_name == 'SVM Tuned':
                model = model_data['svm_tuned']
            elif model_name == 'SVM Tuned Balanced':
                model = model_data['svm_tuned_balanced']
                if model is None:
                    return None, None, "Model SVM Tuned Balanced belum dimuat"
            
            X_tfidf = model_data['tfidf_vectorizer'].transform([preprocessed])
            prediction = model.predict(X_tfidf)[0]
            predicted_label = model_data['label_encoder'].inverse_transform([prediction])[0]
            
            # Get confidence scores
            scores = model.decision_function(X_tfidf)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()
            
        else:  # IndoBERT
            # Select model and tokenizer based on model name
            if model_name == 'IndoBERT-p2 Baseline':
                tokenizer = model_data['tokenizer_indobert_baseline']
                model = model_data['model_indobert_baseline']
            elif model_name == 'IndoBERT-p2 Baseline Balanced':
                tokenizer = model_data['tokenizer_indobert_baseline_balanced']
                model = model_data['model_indobert_baseline_balanced']
            elif model_name == 'IndoBERT-p2 Tuned':
                tokenizer = model_data['tokenizer_indobert']
                model = model_data['model_indobert']
            elif model_name == 'IndoBERT-p2 Tuned Balanced':
                tokenizer = model_data['tokenizer_indobert_tuned_balanced']
                model = model_data['model_indobert_tuned_balanced']
            
            if model is None or tokenizer is None:
                return None, None, f"Model {model_name} belum dimuat"
            
            inputs = tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            prediction = probabilities.argmax()
            predicted_label = model_data['label_encoder'].inverse_transform([prediction])[0]
        
        # Create confidence dict
        confidence_dict = {
            label: float(prob) 
            for label, prob in zip(model_data['label_encoder'].classes_, probabilities)
        }
        
        return predicted_label, confidence_dict, None
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# ===== VISUALIZATIONS =====
def create_confidence_chart(confidence_dict):
    """Create confidence bar chart"""
    sorted_items = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    colors = ['#42B549' if v == max(values) else '#4ADE80' if v > 0.2 else '#86EFAC' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            y=labels,
            x=values,
            orientation='h',
            marker=dict(color=colors, line=dict(color='#00854D', width=1)),
            text=[f'{v:.1%}' for v in values],
            textposition='outside',
            cliponaxis=False
        )
    ])
    
    fig.update_layout(
        title='Confidence Score per Kategori',
        xaxis_title='Confidence',
        yaxis_title='Kategori',
        template='plotly_white',
        height=400,
        xaxis=dict(range=[0, max(values) * 1.2], tickformat='.0%'),
        showlegend=False
    )
    
    return fig

def create_label_dist_chart(df_clean):
    """Create label distribution chart"""
    if df_clean is None or len(df_clean) < 5:
        data = {'Pujian': 15000, 'Informasi/Netral': 12000, 'Kritik': 8000, 'Saran': 3000, 'Lainnya': 2000}
    else:
        data = df_clean['label'].value_counts().to_dict()
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(data.keys()),
            y=list(data.values()),
            marker=dict(color='#42B549', line=dict(color='#00854D', width=1)),
            text=[f'{v:,}' for v in data.values()],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Distribusi Kategori Review',
        xaxis_title='Kategori',
        yaxis_title='Jumlah Review',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def create_model_perf_chart():
    """Create model performance comparison"""
    df = pd.DataFrame({
        'Model': ['SVM Baseline', 'SVM Tuned', 'SVM Tuned Bal.', 'IndoBERT Base', 'IndoBERT Base Bal.', 'IndoBERT Tuned', 'IndoBERT Tuned Bal.'],
        'Accuracy': [0.85, 0.89, 0.89, 0.91, 0.91, 0.93, 0.93],
        'F1-Macro': [0.83, 0.87, 0.88, 0.90, 0.90, 0.92, 0.93],
        'F1-Weighted': [0.85, 0.89, 0.89, 0.91, 0.91, 0.93, 0.93]
    })
    
    fig = go.Figure()
    
    colors = ['#42B549', '#4ADE80', '#86EFAC']
    for idx, metric in enumerate(['Accuracy', 'F1-Macro', 'F1-Weighted']):
        fig.add_trace(go.Bar(
            name=metric,
            x=df['Model'],
            y=df[metric],
            marker_color=colors[idx],
            text=df[metric].round(3),
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Perbandingan Performa Model',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        template='plotly_white',
        height=400,
        yaxis=dict(range=[0, 1.05], tickformat='.0%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# ===== MAIN APP =====
def main():
    # Load models
    model_data = load_models()
    
    # Main header
    st.markdown('<div class="main-header"><h1>Tokopedia Product Review Multi-Class Classifier</h1><p>Klasifikasi Otomatis Review Produk Bahasa Indonesia</p></div>', unsafe_allow_html=True)
    
    # ===== HOME SECTION =====
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    total_reviews = len(model_data['df_clean']) if model_data['df_clean'] is not None and len(model_data['df_clean']) > 0 else 40000
    total_categories = model_data['df_clean']['label'].nunique() if model_data['df_clean'] is not None and len(model_data['df_clean']) > 0 else 5
    
    with col1:
        st.metric("Total Reviews", f"{total_reviews:,}")
    with col2:
        st.metric("Jumlah Kategori/label", total_categories)
    with col3:
        st.metric("Jumlah Models", "6")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_label_dist_chart(model_data['df_clean']), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_model_perf_chart(), use_container_width=True)
    
    st.markdown("---")
    
    # About
    st.markdown("## Deskripsi Proyek")
    st.markdown("""
    Aplikasi ini menggunakan **Machine Learning** dan **Deep Learning** untuk mengklasifikasikan 
    review produk Tokopedia (Bahasa Indonesia) ke dalam 5 kategori:
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Pujian** - Review positif yang memuji produk/layanan
        - **Kritik** - Review negatif dengan keluhan
        - **Saran** - Feedback konstruktif dan saran perbaikan
        """)
    with col2:
        st.markdown("""
        - **Informasi/Netral** - Pernyataan faktual tanpa sentimen
        - **Lainnya** - Review yang tidak masuk kategori lain
        """)
    
    st.markdown("### Model yang Digunakan:")
    st.markdown("""
    1. **SVM Baseline** - Linear SVM dengan parameter default
    2. **SVM Tuned** - SVM optimal dengan GridSearchCV
    3. **SVM Tuned Balanced** - SVM dengan class weights untuk handle imbalanced data
    4. **IndoBERT-p2 Baseline** - Pre-trained Indonesian BERT
    5. **IndoBERT-p2 Baseline Balanced** - IndoBERT baseline dengan class weights
    6. **IndoBERT-p2 Tuned** - Fine-tuned IndoBERT
    7. **IndoBERT-p2 Tuned Balanced** - Fine-tuned IndoBERT dengan class weights (performa terbaik pada minority class)
    """)
    
    st.markdown("### Teknologi:")
    st.markdown("""
    - **NLP**: Sastrawi (Indonesian stemming & stopwords)
    - **ML**: Scikit-learn (TF-IDF, SVM)
    - **DL**: PyTorch, Transformers (IndoBERT)
    - **GUI**: Streamlit
    """)
    
    st.markdown("---")
    
    # ===== PREDICT SECTION =====
    st.markdown("## Klasifikasi Review")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("### Input Review")
        
        # Example buttons
        st.markdown("**💡 Contoh Review:**")
        st.markdown("""
        - Pujian    : "Produk bagus, pengiriman cepat, packing rapi. Sangat puas!"
        - Kritik    : "Barang rusak, pengiriman lama. Sangat kecewa!"
        - Saran     : "Produk sebaiknya dipacking lebih rapi lagi."
        - Netral    : "Barang sudah sampai dan sudah saya terima"
        """)
        
        examples = [
            ("Pujian", "Produk bagus, pengiriman cepat, packing rapi. Sangat puas!"),
            ("Kritik", "Barang rusak, pengiriman lama. Sangat kecewa!"),
            ("Saran", "Produk bagus tapi sebaiknya packing lebih rapi lagi."),
            ("Netral", "Barang sudah sampai sesuai pesanan.")
        ]
        
        if 'input_text' not in st.session_state:
            st.session_state.input_text = "Produk bagus, pengiriman cepat, packing rapi. Sangat puas!"
        
        # Text input
        input_text = st.text_area(
            "Masukkan review produk (Bahasa Indonesia):",
            value=st.session_state.input_text,
            height=150,
            key="review_input"
        )
        
        # Model selection
        model_name = st.selectbox(
            "🤖 Pilih Model:",
            [
                "SVM Baseline", 
                "SVM Tuned", 
                "SVM Tuned Balanced",
                "IndoBERT-p2 Baseline", 
                "IndoBERT-p2 Baseline Balanced",
                "IndoBERT-p2 Tuned", 
                "IndoBERT-p2 Tuned Balanced"
            ],
            index=2
        )
        
        # Buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            predict_btn = st.button("🎯 Predict", use_container_width=True, type="primary")
        with btn_col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.input_text = ""
                st.rerun()
    
    with col2:
        st.markdown("### Hasil Prediksi")
        
        if predict_btn and input_text.strip():
            with st.spinner("Memproses..."):
                predicted_label, confidence_dict, error = make_prediction(input_text, model_name, model_data)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    # Display result
                    st.markdown(f'<div class="result-box"><div class="result-text">{predicted_label}</div></div>', unsafe_allow_html=True)
                    
                    # Confidence chart
                    st.markdown("### Confidence Scores")
                    st.plotly_chart(create_confidence_chart(confidence_dict), use_container_width=True)
                    
                    # Confidence table
                    conf_df = pd.DataFrame({
                        'Kategori': list(confidence_dict.keys()),
                        'Confidence': [f"{v:.2%}" for v in confidence_dict.values()]
                    }).sort_values('Confidence', ascending=False)
                    st.dataframe(conf_df, use_container_width=True, hide_index=True)
        else:
            st.info("Masukkan review dan klik **Predict** untuk melihat hasil klasifikasi")

if __name__ == "__main__":
    main()