import streamlit as st
import numpy as np
import pickle
import re
import plotly.graph_objects as go

# ============================================
# ⚙️ PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Sentiment & Capital Prediction System",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    section[data-testid="stSidebar"] { background-color: #111827; }
    h1, h2, h3, h4, p, label, span { color: #f8fafc !important; }
    .stButton>button {
        background-color: #3b82f6; color: white; font-weight: bold;
        border-radius: 6px; padding: 0.6em 1em; border: none;
    }
    .stButton>button:hover { background-color: #2563eb; color: white; }
    div[data-baseweb="select"] { background-color: #1f2937; }
    .stTextArea textarea { background-color: #1f2937; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h2 style='color:#38bdf8;'>🧠 MULTI-MODEL SENTIMENT & CAPITAL INTEL SYSTEM</h2>",
    unsafe_allow_html=True
)

# ============================================
# 📂 1. LOAD ALL MODELS (cached so this runs once)
# ============================================
@st.cache_resource(show_spinner="Loading AI inference pipeline...")
def load_all_models():
    with open('sentiment_lr_model.pkl', 'rb') as f:
        lr_sentiment_data = pickle.load(f)
    with open('capital_lr_model.pkl', 'rb') as f:
        lr_capital_data = pickle.load(f)

    with open('sentiment_svm_model.pkl', 'rb') as f:
        svm_sentiment_data = pickle.load(f)
    with open('capital_svm_model.pkl', 'rb') as f:
        svm_capital_data = pickle.load(f)

    with open('federated_sentiment_model.pkl', 'rb') as f:
        fed_sentiment_data = pickle.load(f)
    with open('federated_capital_model.pkl', 'rb') as f:
        fed_capital_data = pickle.load(f)

    return {
        "lr_sent": lr_sentiment_data,       # keys: model, encoder, tfidf_params
        "lr_cap": lr_capital_data,          # keys: model, encoder, tfidf_params
        "svm_sent": svm_sentiment_data,     # keys: model, encoder, type
        "svm_cap": svm_capital_data,        # keys: model, encoder, type
        "fed_sent": fed_sentiment_data,     # keys: model, vectorizer, encoder
        "fed_cap": fed_capital_data,        # keys: model, vectorizer, encoder
    }

# ============================================
# ⚙️ 2. PREPROCESSING
# ============================================
negation_words = {
    "not", "no", "never", "non", "nor", "cannot", "can't", "couldn't", "couldnt",
    "wouldn't", "wouldnt", "shouldn't", "shouldnt", "mightn't", "mightnt", "mustn't",
    "mustnt", "don't", "dont", "doesn't", "doesnt", "isn't", "isnt", "aren't", "arent",
    "hasn't", "hasnt", "haven't", "havent", "didn't", "didnt", "wasn't", "wasnt",
    "weren't", "werent", "neither", "nowhere", "nothing", "nobody", "none", "unable",
    "unwilling", "unlikely", "incapable"
}

def clean_text_ensemble(text):
    """LR + SVM negation parser."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s']", ' ', text)
    words = text.split()
    processed_words = []
    negation_count = 0
    for word in words:
        if word in negation_words:
            negation_count += 1
            continue
        if negation_count % 2 == 1:
            processed_words.append(f"NOT_{word}")
        else:
            processed_words.append(word)
    return " ".join(processed_words)

def clean_text_basic(text):
    """Light cleanup for the federated (SGD) models."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def svm_to_probabilities(decision_values, n_classes=2):
    decision_values = np.array(decision_values).flatten()
    if n_classes == 2:
        prob_positive = 1 / (1 + np.exp(-decision_values[0]))
        return np.array([1 - prob_positive, prob_positive])
    else:
        exp_vals = np.exp(decision_values - np.max(decision_values))
        return exp_vals / np.sum(exp_vals)

# ============================================
# 🧠 3. INFERENCE ENGINES
# ============================================
def run_ensemble_prediction(text, m):
    """LR + SVM Ensemble — both pipelines take raw text directly."""
    cleaned = clean_text_ensemble(text)

    lr_sent_proba = m["lr_sent"]["model"].predict_proba([cleaned])[0]
    svm_sent_decision = m["svm_sent"]["model"].decision_function([cleaned])
    n_sent_classes = len(m["lr_sent"]["encoder"].classes_)
    svm_sent_proba = svm_to_probabilities(svm_sent_decision, n_classes=n_sent_classes)
    ensemble_sent_proba = (lr_sent_proba + svm_sent_proba) / 2

    sent_idx = np.argmax(ensemble_sent_proba)
    sentiment = m["lr_sent"]["encoder"].inverse_transform([sent_idx])[0]
    sentiment_conf = float(ensemble_sent_proba[sent_idx])

    lr_cap_proba = m["lr_cap"]["model"].predict_proba([cleaned])[0]
    svm_cap_decision = m["svm_cap"]["model"].decision_function([cleaned])
    n_cap_classes = len(m["lr_cap"]["encoder"].classes_)
    svm_cap_proba = svm_to_probabilities(svm_cap_decision[0], n_classes=n_cap_classes)
    ensemble_cap_proba = (lr_cap_proba + svm_cap_proba) / 2

    all_capitals = [(m["lr_cap"]["encoder"].inverse_transform([i])[0], prob)
                     for i, prob in enumerate(ensemble_cap_proba)]
    all_capitals.sort(key=lambda x: x[1], reverse=True)
    return cleaned, sentiment, sentiment_conf, all_capitals[:7]

def run_federated_prediction(text, m):
    """Federated (SGD) model — vectorizer is separate from the classifier."""
    cleaned = clean_text_basic(text)

    sent_vec = m["fed_sent"]["vectorizer"].transform([cleaned])
    sent_proba = m["fed_sent"]["model"].predict_proba(sent_vec)[0]
    sent_idx = np.argmax(sent_proba)
    sentiment = m["fed_sent"]["encoder"].inverse_transform([sent_idx])[0]
    sentiment_conf = float(sent_proba[sent_idx])

    cap_vec = m["fed_cap"]["vectorizer"].transform([cleaned])
    cap_proba = m["fed_cap"]["model"].predict_proba(cap_vec)[0]
    all_capitals = [(m["fed_cap"]["encoder"].inverse_transform([i])[0], float(prob))
                     for i, prob in enumerate(cap_proba)]
    all_capitals.sort(key=lambda x: x[1], reverse=True)
    return cleaned, sentiment, sentiment_conf, all_capitals[:7]

# ============================================
# 🖥️ 4. LAYOUT
# ============================================
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown("#### SELECT AI ENGINE MODEL")
    engine = st.selectbox(
        "engine_select",
        ["LR + SVM Ensemble", "Federated Model (SGD)"],
        label_visibility="collapsed"
    )

    st.markdown("#### INPUT RAW TEXT CONTEXT")
    text_input = st.text_area("text_input", height=140, label_visibility="collapsed")

    run_clicked = st.button("EXECUTE INFERENCE ENGINE", use_container_width=True)

    st.markdown("###### PARSED ENGINE OUTPUT TEXT")
    clean_box = st.empty()

    st.markdown("---")
    st.markdown("#### ENGINE SENTIMENT PROJECTION")
    sent_display = st.empty()
    conf_display = st.empty()

with right:
    gauge_display = st.empty()
    st.markdown("#### TOP 7 PREDICTED CAPITALS")
    list_display = st.empty()
    chart_display = st.empty()

# ============================================
# 🚀 RUN INFERENCE ON BUTTON CLICK
# ============================================
if run_clicked:
    if not text_input.strip():
        st.warning("Context string field is empty.")
    else:
        try:
            models = load_all_models()
        except FileNotFoundError as e:
            st.error(
                f"Model file not found: {e.filename}. "
                "Make sure all .pkl model files are in the app's root folder."
            )
            st.stop()

        if engine == "LR + SVM Ensemble":
            cleaned, sentiment, conf, top7 = run_ensemble_prediction(text_input, models)
            bar_color = "#38bdf8"
        else:
            cleaned, sentiment, conf, top7 = run_federated_prediction(text_input, models)
            bar_color = "#22c55e"

        clean_box.text_area("cleaned", value=cleaned, height=100,
                             label_visibility="collapsed", disabled=True)

        sent_color = "#22c55e" if sentiment.lower() == "positive" else "#e11d48"
        sent_display.markdown(
            f"<h1 style='color:{sent_color}; margin-bottom:0;'>{sentiment.upper()}</h1>",
            unsafe_allow_html=True
        )
        conf_display.progress(conf, text=f"Confidence Matrix: {conf*100:.2f}%")

        angle_value = conf * 100 if sentiment.lower() == "positive" else (1 - conf) * 100
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=angle_value,
            number={'suffix': "%", 'font': {'color': '#f8fafc'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#f8fafc'},
                'bar': {'color': '#38bdf8'},
                'bgcolor': "#1f2937",
                'borderwidth': 0,
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0f172a", font={'color': "#f8fafc"},
            height=220, margin=dict(t=10, b=10, l=20, r=20)
        )
        gauge_display.plotly_chart(fig_gauge, use_container_width=True)

        rows_html = ""
        for name, prob in top7:
            rows_html += f"""
            <div style="display:flex;align-items:center;background:#1f2937;
                        padding:8px 15px;margin:4px 0;border-radius:4px;">
                <div style="width:200px;color:white;font-weight:bold;">{name}</div>
                <div style="flex:1;background:#334155;border-radius:4px;height:14px;margin:0 15px;">
                    <div style="width:{prob*100}%;background:{bar_color};height:14px;border-radius:4px;"></div>
                </div>
                <div style="width:70px;text-align:right;color:{bar_color};font-weight:bold;">{prob*100:.2f}%</div>
            </div>
            """
        list_display.markdown(rows_html, unsafe_allow_html=True)

        keys = [t[0] for t in top7][::-1]
        vals = [t[1] * 100 for t in top7][::-1]
        fig_bar = go.Figure(go.Bar(
            x=vals, y=keys, orientation='h', marker_color=bar_color
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1f2937",
            font={'color': "#f8fafc"}, height=350,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"),
        )
        chart_display.plotly_chart(fig_bar, use_container_width=True)
else:
    clean_box.text_area("cleaned", value="", height=100,
                         label_visibility="collapsed", disabled=True)
    sent_display.markdown(
        "<h1 style='color:#9ca3af; margin-bottom:0;'>---</h1>", unsafe_allow_html=True
    )
    conf_display.progress(0, text="Confidence Matrix: 0.00%")
