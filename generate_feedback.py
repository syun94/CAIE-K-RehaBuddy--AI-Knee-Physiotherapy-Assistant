import os, re, textwrap 
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from getpass import getpass

# =========================
# CONFIG
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = getpass("Enter your OpenAI API key: ").strip()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

KNOWLEDGE_BASE_DIR = "knowledge_base"

# --- posture label definitions ---
WRONG_LABEL_KEYS = [
    "squat_wt", "squat_fl",
    "extension_nf", "extension_ll",
    "gait_nf", "gait_ha"
]
NORMAL_LABEL_KEYS = ["squat", "extension", "gait"]

# --- guideline mapping ---
GUIDELINE_MAP = {
    "squat": "doc_01_squat_guideline.txt",
    "extension": "doc_02_extensions_guideline.txt",
    "gait": "doc_04_gait_guideline.txt"
}

# =========================
# CLIENT SETUP
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
chroma_client = chromadb.Client()

COLLECTION_NAME = "posture_knowledge"

# =========================
# COLLECTION INIT
# =========================
try:
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
except Exception:
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"⚠️ Rebuilding Chroma collection '{COLLECTION_NAME}'...")
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

# =========================
# LOAD KNOWLEDGE BASE
# =========================
existing_docs = collection.get()
if not existing_docs["ids"]:
    docs, ids = [], []
    for i, file in enumerate(sorted(os.listdir(KNOWLEDGE_BASE_DIR))):
        if file.endswith(".txt"):
            with open(os.path.join(KNOWLEDGE_BASE_DIR, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
                ids.append(f"doc_{i+1}")
    if docs:
        collection.add(documents=docs, ids=ids)
        print(f"✅ Added {len(docs)} documents to '{collection.name}'")

# =========================
# MAIN FEEDBACK FUNCTION
# =========================
def generate_posture_feedback(predicted_label: str, confidence: float, query_text: str, collection):
    """Return natural conversational feedback for a predicted posture label."""

    # --- retrieve relevant guidelines automatically ---
    exercise_type = None
    for key in GUIDELINE_MAP.keys():
        if key in predicted_label.lower():
            exercise_type = key
            break

    guideline_text = ""
    if exercise_type:
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, GUIDELINE_MAP[exercise_type])
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                guideline_text = f.read().strip()

    # --- retrieve from vector db as secondary context ---
    try:
        res = collection.query(query_texts=[query_text], n_results=2)
        retrieved_docs = [d for docs in res["documents"] for d in docs]
    except Exception as e:
        print(f"⚠️ Reinitializing collection due to error: {e}")
        chroma_client.delete_collection(name=COLLECTION_NAME)
        collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
        for i, file in enumerate(sorted(os.listdir(KNOWLEDGE_BASE_DIR))):
            if file.endswith(".txt"):
                with open(os.path.join(KNOWLEDGE_BASE_DIR, file), "r", encoding="utf-8") as f:
                    collection.add(documents=[f.read()], ids=[f"doc_{i+1}"])
        res = collection.query(query_texts=[query_text], n_results=2)
        retrieved_docs = [d for docs in res["documents"] for d in docs]

    # --- add wrong-posture definitions if needed ---
    if any(k in predicted_label.lower() for k in WRONG_LABEL_KEYS):
        wrong_doc_path = os.path.join(KNOWLEDGE_BASE_DIR, "doc_03_definitions_wrong_postures_during_exercise.txt")
        if os.path.exists(wrong_doc_path):
            with open(wrong_doc_path, "r", encoding="utf-8") as f:
                wrong_text = f.read().strip()
            retrieved_docs.insert(0, wrong_text)

    # --- merge all context ---
    context_parts = []
    if guideline_text:
        context_parts.append(f"📘 Official {exercise_type.capitalize()} guideline:\n{guideline_text}")
    if retrieved_docs:
        context_parts.append("\n".join(retrieved_docs))
    context = "\n\n".join(context_parts)

    # -----------------------------
    # System + user messages
    # -----------------------------
    system_msg = textwrap.dedent(f"""
    You are a kind, supportive physiotherapy assistant called KRehaBuddy.
    You respond naturally like texting a patient.
    Use the provided context (official exercise guidelines and posture definitions) to give advice.

    Rules:
    - WRONG posture labels: {WRONG_LABEL_KEYS}
    - NORMAL posture labels: {NORMAL_LABEL_KEYS}
    - If the label is WRONG → use the guidelines to give **constructive correction** with practical cues (body position, balance, or movement advice) and gentle encouragement.
    - If the label is NORMAL → use the guidelines to give **positive reinforcement** (acknowledge good form, rhythm, or stability) and one short safety reminder.
    - Always be friendly and concise (2–4 sentences).
    - Never contradict the prediction — if it’s normal, don’t suggest it’s wrong.
    - Avoid technical jargon unless explained simply.
    """).strip()

    user_msg = f"""
    The ML model predicted label: {predicted_label} (confidence {confidence:.2f})

    Relevant exercise type: {exercise_type}
    Context and reference material:
    {context}
    """

    # -----------------------------
    # Call LLM
    # -----------------------------
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        reply = resp.choices[0].message.content.strip()
        return reply

    except Exception as e:
        print(f"LLM fallback due to: {e}")
        # Simple fallback message
        if any(k in predicted_label.lower() for k in WRONG_LABEL_KEYS):
            return "Try to correct your form a bit — small adjustments go a long way. Keep your movements smooth and controlled!"
        else:
            return "Excellent work! Your form looks consistent and stable. Keep moving with control and steady breathing."
