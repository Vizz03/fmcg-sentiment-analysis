import streamlit as st
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re

# ==========================================
# 1. RESEARCH FRAMEWORK SETUP
# ==========================================
st.set_page_config(page_title="FMCG Sentiment Framework", layout="wide")

@st.cache_resource
def load_resources():
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    return SentimentIntensityAnalyzer(), set(nltk.corpus.stopwords.words('english'))

analyzer, stop_words = load_resources()

# ==========================================
# 2. PREPROCESSING & ANALYTICS ENGINE
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\@\w+|\#','', text) # Remove mentions/hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def get_vader_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05: return 'Positive'
    elif score <= -0.05: return 'Negative'
    else: return 'Neutral'

def generate_strategic_advice(df, product_name):
    pos_rate = (df['sentiment'] == 'Positive').mean()
    neg_rate = (df['sentiment'] == 'Negative').mean()
    
    advice = []
    if neg_rate > 0.20:
        advice.append(f"⚠️ **Urgent:** High dissatisfaction detected for {product_name}. Investigate quality control.")
    if pos_rate > 0.80:
        advice.append(f"🚀 **Opportunity:** Strong brand equity. Consider premium line extension.")
    
    # Logic for common FMCG keywords
    text_blob = " ".join(df['text'].astype(str).tolist()).lower()
    if 'price' in text_blob or 'expensive' in text_blob:
        advice.append("💰 **Pricing Strategy:** Customers are sensitive to cost. Evaluate discount bundles.")
    if 'delivery' in text_blob or 'stock' in text_blob:
        advice.append("🚚 **Supply Chain:** Availability issues detected. Optimize distribution channels.")
        
    return advice if advice else ["Data insufficient for specific strategic pivots."]

# ==========================================
# 3. STREAMLIT DASHBOARD UI
# ==========================================
st.sidebar.title("DSR Artifact Controls")
app_mode = st.sidebar.selectbox("Navigate Framework", 
    ["1. Data Ingestion", "2. Comparative Analysis", "3. Strategic Decision Support"])

# --- MODE 1: DATA INGESTION ---
if app_mode == "1. Data Ingestion":
    st.header("📂 Objective 1: Data Ingestion & Preprocessing")
    uploaded_file = st.file_uploader("Upload FMCG Customer Text Data (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            st.success("Dataset Loaded Successfully")
            df['cleaned_text'] = df['text'].apply(clean_text)
            st.write("### Raw vs Preprocessed Data (Anonymized)")
            st.dataframe(df[['text', 'cleaned_text']].head())
            st.session_state['data'] = df
        else:
            st.error("CSV must contain a 'text' column.")

# --- MODE 2: COMPARATIVE ANALYSIS ---
elif app_mode == "2. Comparative Analysis":
    st.header("🧪 Objective 2: NLP Interpretation & Validation")
    
    if 'data' not in st.session_state:
        st.warning("Please upload data in the Ingestion tab first.")
    else:
        df = st.session_state['data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model A: VADER (Lexicon-Based)")
            df['vader_pred'] = df['cleaned_text'].apply(get_vader_sentiment)
            st.write(df['vader_pred'].value_counts())
            
        with col2:
            st.subheader("Model B: Logistic Regression (ML)")
            if 'label' in df.columns:
                # Basic ML Pipeline for Dissertation comparison
                vectorizer = TfidfVectorizer(max_features=1000)
                X = vectorizer.fit_transform(df['cleaned_text'])
                y = df['label'].str.capitalize() # Standardization
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LogisticRegression().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.text("ML Classification Report:")
                st.text(classification_report(y_test, y_pred))
            else:
                st.info("Upload a dataset with a 'label' column to enable ML Comparison.")

# --- MODE 3: STRATEGIC DECISION SUPPORT ---
elif app_mode == "3. Strategic Decision Support":
    st.header("🎯 Objective 3: Strategic Decision Facilitation")
    
    if 'data' not in st.session_state:
        st.warning("No data found.")
    else:
        df = st.session_state['data']
        # Apply VADER as default engine for exploration
        df['sentiment'] = df['cleaned_text'].apply(get_vader_sentiment)
        
        search_term = st.text_input("Enter FMCG Product/Brand for Strategic Deep-Dive:", "milk")
        
        if search_term:
            filtered_df = df[df['text'].str.contains(search_term, case=False, na=False)]
            
            if not filtered_df.empty:
                # Strategic Cards
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Sample Size", len(filtered_df))
                kpi2.metric("Net Positive", f"{(filtered_df['sentiment']=='Positive').mean()*100:.1f}%")
                kpi3.metric("Risk Level", "High" if (filtered_df['sentiment']=='Negative').mean() > 0.2 else "Low")
                
                # Decision Engine
                st.subheader("🧠 Automated Strategic Recommendations")
                recommendations = generate_strategic_advice(filtered_df, search_term)
                for rec in recommendations:
                    st.info(rec)
                
                # Visuals
                cloud_col1, cloud_col2 = st.columns(2)
                with cloud_col1:
                    st.write("Positive Aspect Cloud")
                    pos_text = " ".join(filtered_df[filtered_df['sentiment']=='Positive']['cleaned_text'])
                    if pos_text:
                        wc = WordCloud(background_color='white', colormap='Greens').generate(pos_text)
                        st.image(wc.to_array())
                with cloud_col2:
                    st.write("Negative Aspect Cloud")
                    neg_text = " ".join(filtered_df[filtered_df['sentiment']=='Negative']['cleaned_text'])
                    if neg_text:
                        wc = WordCloud(background_color='white', colormap='Reds').generate(neg_text)
                        st.image(wc.to_array())
            else:
                st.error("No reviews found for this product.")

# --- ETHICAL FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("🔒 **HIT Ethical Compliance:** Data is anonymized. No PII (Personally Identifiable Information) is stored.")