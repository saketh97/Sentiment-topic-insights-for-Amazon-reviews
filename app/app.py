import os
import pandas as pd
from joblib import load
import streamlit  as st
import numpy as np

# Session defaults to avoid NameError across reruns
if "last_topics" not in st.session_state:
    st.session_state.last_topics = []
if "last_sentiment" not in st.session_state:
    st.session_state.last_sentiment = None

ST_MODELS = {
    "sentiment":"models/baseline_tfidf_logreg.joblib",
    "vect":"models/topics/count_vectorizer.joblib",
    "lda":"models/topics/lda_model.joblib"
}

top_terms_csv = "reports/topics/lda_top_terms.csv"
TOP_LABELS_CSV = "reports/topics/lda_topic_labels.csv"

@st.cache_resource
def load_sentiment_model():
    return load(ST_MODELS["sentiment"]) if os.path.exists(ST_MODELS["sentiment"]) else None

@st.cache_resource
def load_topic_models():
    vect=load(ST_MODELS["vect"]) if os.path.exists(ST_MODELS["vect"]) else None
    lda=load(ST_MODELS["lda"]) if os.path.exists(ST_MODELS["lda"]) else None
    return vect, lda

@st.cache_data
def load_top_terms():
    if os.path.exists(top_terms_csv):
        df = pd.read_csv(top_terms_csv)
        return df
    return pd.DataFrame(columns=["topic","rank","term"])

@st.cache_data
def load_topic_labels():
    # If labels file exists, use it; else derive from top terms
    if os.path.exists(TOP_LABELS_CSV):
        df = pd.read_csv(TOP_LABELS_CSV)
        return {int(r.topic): str(r.label) for _, r in df.iterrows()}
    # fallback: build simple labels from top terms
    df_terms = load_top_terms()
    label_map = {}
    if not df_terms.empty:
        for k in sorted(df_terms["topic"].unique()):
            top3 = (df_terms[df_terms["topic"]==k]
                    .sort_values("rank")
                    .head(3)["term"].tolist())
            label_map[int(k)] = " / ".join(top3) if top3 else f"Topic {k}"
    return label_map


def predict_sentiment(pipe, text):
    if not pipe or not text.strip():
        return None
    proba=None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba([text])[0][1]
    pred = pipe.predict([text])[0]
    return int(pred), (float(proba) if proba is not None else None)

def topic_distrbution(vect, lda, text, top_k=3):
    if not (vect and lda) or not text.strip():
        return []
    X=vect.transform([text])
    dist=lda.transform(X)[0]
    idx = np.argsort(dist)[-top_k:][::-1]
    return list(zip(idx,dist[idx]))

st.set_page_config(page_title="Marketing NLP Assistant",layout="wide")
st.title("Marketing review assistant")
st.caption("Live sentiment + topic peek on amazon-style reviews")

label_map = load_topic_labels()
df_terms = load_top_terms()

left,right = st.columns([2,1])

with left:
    text = st.text_area("Enter a review text", height=160,placeholder="e.g.,he sound quality is amazing but the strap broke in a week. ") 
    if st.button("Analyze"):
        pipe = load_sentiment_model()
        pred = predict_sentiment(pipe, text)
        if pred is None:
            st.warning("Sentiment model not found. Train baseline first.")
            st.session_state.last_sentiment = None
        else:
            label, proba = pred
            lbl = "Positive" if label == 1 else "Negative"
            st.markdown(f"**Sentiment:** {lbl}")
            if proba is not None:
                st.markdown(f"**Confidence (Positive):** {proba:.3f}")
            st.session_state.last_sentiment = {"label": lbl, "proba": proba}

        vect, lda = load_topic_models()
        computed = topic_distrbution(vect, lda, text, top_k=3)
        st.session_state.last_topics = computed

    # Show “Likely topics” even after reruns (uses last saved result)
    if st.session_state.last_topics:
        st.markdown("**Likely topics:**")
        for tid, score in st.session_state.last_topics:
            name = label_map.get(int(tid), f"Topic {tid}")
            st.write(f"- **{name}** — Relevance: {score:.3f}")

with right:
    st.subheader("Top terms")
    if df_terms.empty:
        st.info("Run topic training to see top terms.")
    else:
        # Show only predicted topics first
        if st.session_state.last_topics:
            st.markdown("**Predicted topics (terms):**")
            for tid, _ in st.session_state.last_topics:
                terms = (df_terms[df_terms["topic"] == int(tid)]
                         .sort_values("rank").head(8)["term"].tolist())
                name = label_map.get(int(tid), f"Topic {tid}")
                st.markdown(f"- **{name}**: " + ", ".join(terms) if terms else f"- **{name}** (no terms)")
        # Browse all topics in an expander
        with st.expander("All topics (browse)"):
            for k in sorted(df_terms["topic"].unique()):
                terms = (df_terms[df_terms["topic"] == k]
                         .sort_values("rank").head(8)["term"].tolist())
                name = label_map.get(int(k), f"Topic {k}")
                st.markdown(f"- **{name}**: " + ", ".join(terms))

