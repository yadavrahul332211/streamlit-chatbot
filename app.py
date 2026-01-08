import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chatbot App", page_icon="🤖")

st.title("🤖 NLP Chatbot")
st.write("Question pucho, chatbot answer dega")

# Load data
df = pd.read_csv("qa_data.csv")
questions = df["question"].str.lower().tolist()
answers = df["answer"].tolist()

# TF-IDF
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # NLP similarity
    user_vector = vectorizer.transform([user_input.lower()])
    similarity = cosine_similarity(user_vector, question_vectors)

    best_index = similarity.argmax()
    best_score = similarity[0][best_index]

    if best_score > 0.3:
        bot_reply = answers[best_index]
    else:
        bot_reply = "Sorry 😕 mujhe is question ka answer nahi pata."

    # Bot message
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )

    st.rerun()

