import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

st.set_page_config(
    page_title="FlipItNews Classifier",
    page_icon="📰",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "news_rnn.keras")
    )

    with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(BASE_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    return model, tokenizer, encoder, metadata


model, tokenizer, encoder, metadata = load_artifacts()

st.title("📰 FlipItNews — AI News Classifier")
st.caption(
    "Automatically categorize news articles using a trained recurrent "
    "neural network with Word2Vec embeddings."
)

with st.sidebar:
    st.header("About the project")
    st.write("**Model:**", metadata["model_name"])
    st.write("**Categories:**", len(metadata["class_names"]))
    st.write("**Sequence length:**", metadata["max_len"])
    st.markdown("---")
    st.write("Enter a news headline and article to get a prediction.")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()

    # Keep the same stop-word policy used in training.
    # Tokenizer handles out-of-vocabulary words.
    return " ".join(tokens)

headline = st.text_input(
    "News headline",
    placeholder="Enter the headline here..."
)

article = st.text_area(
    "News article",
    height=250,
    placeholder="Paste the full news article here..."
)

if st.button("Classify News", type="primary"):
    combined_text = (headline.strip() + " " + article.strip()).strip()

    if not combined_text:
        st.warning("Please enter a headline or article.")
    else:
        clean_text = preprocess_text(combined_text)
        sequence = tokenizer.texts_to_sequences([clean_text])

        padded = pad_sequences(
            sequence,
            maxlen=metadata["max_len"],
            padding="post",
            truncating="post"
        )

        probabilities = model.predict(padded, verbose=0)[0]
        best_idx = int(np.argmax(probabilities))
        predicted_category = encoder.inverse_transform([best_idx])[0]
        confidence = float(probabilities[best_idx])

        st.subheader("Classification result")
        st.success(f"Predicted category: **{predicted_category}**")
        st.metric("Model confidence", f"{confidence:.2%}")

        scores_df = pd.DataFrame({
            "Category": encoder.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.subheader("Category probability scores")
        st.bar_chart(scores_df.set_index("Category"))

        st.dataframe(
            scores_df.assign(
                Probability=scores_df["Probability"].map(
                    lambda value: f"{value:.2%}"
                )
            ),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Confidence is the model's predicted probability, not a guarantee "
            "that the category is correct. Review low-confidence predictions."
        )

st.markdown("---")
st.caption(
    "FlipItNews NLP project | Bag of Words • TF-IDF • Word2Vec • RNN"
)
