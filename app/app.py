import streamlit as st
import pandas as pd

# Load data
movies = pd.read_csv("movies_clean.csv")

# -------------------------
# IMPORTANT: Load your model data
# -------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Prepare data
content_df = movies[['movieId', 'title', 'genres']].drop_duplicates()
content_df['genres'] = content_df['genres'].fillna("").astype(str)

# Convert genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(content_df['genres'])

# Similarity
indices = pd.Series(content_df.index, index=content_df['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title, top_n=10):
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[::-1][1:top_n+1]

    return content_df['title'].iloc[sim_indices]

# -------------------------
# STREAMLIT UI
# -------------------------

st.title("🎬 Movie Recommendation System")

movie = st.selectbox("Choose a movie", content_df['title'])

if st.button("Recommend"):
    recommendations = get_recommendations(movie)

    if len(recommendations) == 0:
        st.write("Movie not found.")
    else:
        st.subheader("Recommended Movies:")
        for rec in recommendations:
            st.write(rec)