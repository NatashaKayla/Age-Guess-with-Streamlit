# 🧒👩‍💼 Age Guess Based on Chat Responses

Predict a user's **age group** from **five short text answers**.  
This project covers the full workflow: text preprocessing, model training (TF-IDF + classifier), packaging the best **scikit-learn Pipeline**, and a Streamlit app for interactive prediction.

> Default labels: **Gen Alpha, Gen Z, Millennial, Gen X, Boomer**  
> The app reads the actual class names from the saved model.

---

## 🎯 Objectives

- Collect 5 chat-style answers and predict a single age group  
- Build a robust text-classification pipeline (TF-IDF + linear model)  
- Provide a public, no-backend UI with **Streamlit**  
- Keep deployment simple and reproducible (GitHub + Streamlit Cloud)

---

## 🧾 Dataset & Prompts

**Five questions used by the app:**

1) *How do you usually greet your friends in chat?*  
2) *What kind of music do you listen to most?*  
3) *What apps do you use every day?*  
4) *How do you usually spend your weekends?*  
5) *What’s your favorite way to spend free time?*

**Target labels (example):** `Gen Alpha`, `Gen Z`, `Millennial`, `Gen X`, `Boomer`.

---

## 🧹 Preprocessing (training & inference)

- Keep emojis (when the model is trained with emojis).  
- Lowercase + light punctuation/space normalization.  
- **Answer weighting:** each answer is repeated **×3** inside the `text` feature (matches training).  
- No leakage: `age_group` is **only** the label, never included in the text.

---

## 🧠 Modeling

- **Features:** `TfidfVectorizer` with n-grams (1,2), `min_df`/`max_df` tuned.  
- **Models tried:** Logistic Regression / LinearSVC / SGDClassifier / Naive Bayes (grid search with stratified CV, scored by **macro-F1**).  
- **Selected model:** best CV macro-F1 on the train split.  
- **Saved artifact:** the **entire scikit-learn Pipeline** (TF-IDF + classifier) via `joblib.dump(...)`.

---

## 🖥 Streamlit App (inference)

- Two-column form: **left = question**, **right = answer**.  
- Sends the 5 answers through the saved Pipeline.  
- For each answer, get the probability distribution across classes (or decision scores → softmax).  
- **Final decision:** **average the per-class probabilities** across the 5 answers, then pick the top class.  
  - Fallback: **majority vote** if probabilities are not available.  
- Optional UI details: per-question Top-1 confidence and a bar chart of averaged probabilities; “uncertain” warning when max averaged prob < threshold.

---

## 🔍 Evaluation (training notebook)

Typical metrics to report:

* **Accuracy** and **Macro-F1** on a hold-out test set
* **Classification report** per class
* **Confusion matrix** (counts + row-normalized)

---

## 🚀 Live Demo

Try the app here: https://age-guess-with-app.streamlit.app/
