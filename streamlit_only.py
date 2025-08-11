import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re

st.set_page_config(page_title="Age Guess Based on Chat Responses", layout="centered")
st.title("🧒👩‍💼 Age Guess Based on Chat Responses")
st.caption("Predict the age group from five short text responses.")

# ---------------- Config ----------------
MODEL_PATH = "models/age_guess_best_pipeline.joblib"

# Your exact 5 questions
QUESTIONS = [
    "How do you usually greet your friends in chat?",
    "What kind of music do you listen to most?",
    "What apps do you use every day?",
    "How do you usually spend your weekends?",
    "What's your favorite way to spend free time?"
]

# Since the model was trained WITH emojis, we DO NOT remove emojis here.
def build_text(answer: str) -> str:
    """Replicate training: keep emojis, repeat answer x3, normalize."""
    a = "" if answer is None else str(answer)
    txt = (a + ", ") * 3
    txt = re.sub(r"\s+", " ", txt.lower())
    txt = re.sub(r"(,?\s*,\s*)+", ", ", txt).strip(", ").strip()
    return txt

@st.cache_resource
def load_model():
    pipe = joblib.load(MODEL_PATH)  # entire Pipeline: TF-IDF + classifier
    last = list(pipe.named_steps.keys())[-1]
    classes = pipe.named_steps[last].classes_.tolist()
    return pipe, classes

pipe, CLASSES = load_model()
st.sidebar.success("Model loaded")
st.sidebar.caption("Classes: " + ", ".join(CLASSES))

def topk_probs(texts, k=None):
    """Return per-item list of (label, prob). Works with predict_proba or decision_function fallback."""
    last = list(pipe.named_steps.keys())[-1]
    k = len(CLASSES) if (k is None or k > len(CLASSES)) else k
    clf = pipe.named_steps[last]

    if hasattr(clf, "predict_proba"):
        proba = pipe.predict_proba(texts)
    elif hasattr(clf, "decision_function"):
        scores = pipe.decision_function(texts)
        if scores.ndim == 1:  # binary margin
            scores = np.vstack([-scores, scores]).T
        scores = scores - scores.max(axis=1, keepdims=True)
        proba = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    else:
        preds = pipe.predict(texts)
        return [[(preds[i], 1.0)] for i in range(len(texts))]

    out = []
    for row in proba:
        idx = np.argsort(-row)[:k]
        out.append([(CLASSES[i], float(row[i])) for i in idx])
    return out

def average_probs(topk_list):
    """Average per-class probabilities across all 5 answers."""
    agg = {c: [] for c in CLASSES}
    for tk in topk_list:
        m = {c: p for c, p in tk}
        for c in CLASSES:
            agg[c].append(m.get(c, 0.0))
    return {c: float(np.mean(v)) for c, v in agg.items()}

# --------------- Two-column form ---------------
st.markdown("Answer each question (right column), then click **Predict**.")
with st.form("age_guess_form"):
    answers = []
    for i, q in enumerate(QUESTIONS, start=1):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"**Question {i}.** {q}")
        with c2:
            a = st.text_input("Answer", key=f"a{i}", placeholder="Type your answer…", label_visibility="collapsed")
        answers.append(a)
    submitted = st.form_submit_button("🎯 Predict Age Group")

if submitted:
    if any(not (a and a.strip()) for a in answers):
        st.warning("Please fill in all 5 answers.")
    else:
        texts = [build_text(a) for a in answers]       # keep emojis
        tk = topk_probs(texts, k=len(CLASSES))

        # Per-question table
        rows = []
        for i, tk_i in enumerate(tk, 1):
            pred_label, pred_prob = tk_i[0]
            rows.append({"Question": i, "Prediction": pred_label})
        st.subheader("Per-question predictions")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Final decision: average probabilities; fallback to majority vote
        try:
            avg = average_probs(tk)
            final_label = max(avg, key=avg.get)
            st.success(f"🏁 Final Age Group: **{final_label}**")
        except Exception:
            preds = [row["Prediction"] for row in rows]
            final_label = pd.Series(preds).mode().iloc[0]
            st.info("Probabilities unavailable → using majority vote.")
            st.success(f"🏁 Final Age Group: **{final_label}**")