# --- app.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import numpy as np
import chromadb
from pathlib import Path
from generate_feedback import generate_posture_feedback
from config import MODEL_PATH
from openai import OpenAI
import os

# ------------------------------
# 1️⃣ Model Definition
# ------------------------------
class CNN1D_AttentionClassifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25)
        )

        self.attention = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, 1, kernel_size=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        attn_weights = self.attention(feat)
        attn_weights = F.softmax(attn_weights, dim=-1)
        context = torch.sum(feat * attn_weights, dim=-1)
        out = self.classifier(context)
        return out


# ------------------------------
# 2️⃣ Load Model
# ------------------------------
@st.cache_resource
def load_model(model_path=MODEL_PATH, device="cpu"):
    model = CNN1D_AttentionClassifier(input_channels=56, num_classes=9)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


model, device = load_model()

# ------------------------------
# 3️⃣ Load Chroma Collection
# ------------------------------
@st.cache_resource
def load_collection():
    client = chromadb.Client()
    try:
        return client.get_or_create_collection("krehabuddy_feedback")
    except Exception as e:
        st.warning(f"⚠️ ChromaDB issue detected: {e}. Reinitializing collection.")
        return client.create_collection("krehabuddy_feedback")


collection = load_collection()

# ------------------------------
# 4️⃣ LLM Setup
# ------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------
# 5️⃣ Label Map
# ------------------------------
LABEL_MAP = {
    0: "Squat",
    1: "Squat_WT",
    2: "Squat_FL",
    3: "Extension",
    4: "Extension_NF",
    5: "Extension_LL",
    6: "Gait",
    7: "Gait_NF",
    8: "Gait_HA"
}


# ------------------------------
# 6️⃣ Inference Helper
# ------------------------------
def run_inference(npz_file):
    data = np.load(npz_file)
    X = data["X"]
    y_true = int(data["y"]) if "y" in data else None

    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(X_t)
        probs = torch.softmax(out, dim=1)
        conf, pred_class = torch.max(probs, dim=1)

    pred_label = LABEL_MAP[int(pred_class)]
    print("\n========== DEBUG: MODEL INFERENCE ==========")
    print(f"Predicted class index: {int(pred_class)}")
    print(f"Predicted label: {pred_label}")
    print(f"Confidence: {float(conf):.4f}")
    print("===========================================\n")
    return pred_label, float(conf), y_true


# ------------------------------
# 7️⃣ LLM QA Helper (for user questions)
# ------------------------------
def answer_user_question(question: str):
    """Retrieve relevant docs and use GPT to answer user’s custom question."""
    res = collection.query(query_texts=[question], n_results=3)
    retrieved = [d for docs in res["documents"] for d in docs]
    context = "\n\n".join(retrieved)

    system_msg = (
        "You are K-RehaBuddy, a friendly physiotherapy assistant. "
        "Answer clearly and kindly based on the provided exercise and knee rehabilitation guidelines. "
        "Do not guess; if unsure, say you’re not certain and suggest consulting a physiotherapist."
    )

    user_msg = f"User question: {question}\n\nContext:\n{context}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.6,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


# ------------------------------
# 8️⃣ Streamlit Chat Flow
# ------------------------------
st.set_page_config(page_title="K-RehaBuddy", page_icon="🧍‍♀️")
st.title("🧍‍♀️ K-RehaBuddy: Your Physiotherapy Assistant")

if "stage" not in st.session_state:
    st.session_state.stage = "greeting"

# --- Stage 1: Greeting ---
if st.session_state.stage == "greeting":
    st.write("👋 Hi! I’m **K-RehaBuddy**, your virtual physiotherapist. Upload your exercise data file (`.npz`) to get posture feedback.")
    uploaded = st.file_uploader("Upload your exercise file", type=["npz"])

    if uploaded:
        st.session_state.uploaded = uploaded
        st.session_state.stage = "feedback"
        st.success("✅ File uploaded successfully!")
        st.rerun()


# --- Stage 2: Model Feedback ---
elif st.session_state.stage == "feedback":
    st.success("✅ File uploaded successfully!")
    uploaded = st.session_state.uploaded

    with st.spinner("Analyzing your exercise posture..."):
        predicted_label, confidence, _ = run_inference(uploaded)
        feedback = generate_posture_feedback(predicted_label, confidence, "posture analysis", collection)

    st.subheader("🧠 Posture Feedback")
    st.markdown(feedback)

    st.session_state.predicted_label = predicted_label
    st.session_state.confidence = confidence
    st.session_state.stage = "pain_question"

    if st.button("Next ➡️"):
        st.rerun()


# --- Stage 3: Pain Level Inquiry ---
elif st.session_state.stage == "pain_question":
    st.subheader("💬 Let’s check your comfort level")
    pain_level = st.slider("How would you rate your pain during this exercise? (0 = no pain, 10 = worst pain)", 0, 10, 3)

    if st.button("Submit Pain Level"):
        st.session_state.pain_level = pain_level
        st.session_state.stage = "pain_feedback"
        st.rerun()


# --- Stage 4: Pain Feedback ---
elif st.session_state.stage == "pain_feedback":
    pain_level = st.session_state.pain_level

    with open("knowledge_base/doc_05_How-to-tell-if-you-are-exercising-at-right-level.txt", "r", encoding="utf-8") as f:
        pain_doc = f.read()

    st.subheader("🩺 Pain Level Feedback")

    if pain_level <= 3:
        st.success("✅ Minimal pain — you’re exercising safely. Keep up the good work!")
    elif pain_level <= 5:
        st.warning("⚠️ Mild discomfort — acceptable, but monitor your form and stop if it worsens.")
    else:
        st.error("🚨 Too much pain — stop immediately, rest, and reduce intensity next time. Consult your physiotherapist if it persists.")

    st.markdown("---")

    # Initialize chat state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_round" not in st.session_state:
        st.session_state.chat_round = 0
    if "chat_active" not in st.session_state:
        st.session_state.chat_active = True

    # Rephrased prompt per round
    followup_prompts = [
        "Do you have any question about exercising for your knee?",
        "Anything else you’d like to ask about your exercises?",
        "Would you like to know more about another exercise or movement?",
        "Is there something else you'd like me to explain or clarify?",
        "Any other question before we wrap up?"
    ]

    current_prompt = (
        followup_prompts[min(st.session_state.chat_round, len(followup_prompts) - 1)]
        if st.session_state.chat_active
        else None
    )

    # Active chat flow
    if st.session_state.chat_active:
        st.markdown(f"💬 {current_prompt}")
        user_q = st.text_input("Type your question here 👇", key=f"user_q_{len(st.session_state.chat_history)}")

        if user_q:
            lower_q = user_q.strip().lower()

            # --- Check for exit intent ---
            if lower_q in ["no", "no thanks", "not now", "that's all", "nothing", "nah"]:
                st.session_state.chat_active = False
                st.session_state.farewell = (
                    "🌟 Great work today! Keep listening to your body, stay consistent, "
                    "and you’ll see steady progress. Take care and see you next session!"
                )
                st.rerun()

            else:
                with st.spinner("Thinking..."):
                    answer = answer_user_question(user_q)  # <- this calls your RAG/LLM function
                st.session_state.chat_history.append((user_q, answer))
                st.session_state.chat_round += 1
                st.rerun()

    # Display chat history
    if "chat_history" in st.session_state:
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**K-RehaBuddy:** {a}")
            st.markdown("---")

    # Farewell when user says no
    if not st.session_state.get("chat_active", True):
        st.success(st.session_state.get("farewell", "😊 Take care!"))

