import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

API_KEY = "022b877171805be41e9b2ffd24e5ded3"

# -------------------------
# PAGE SETUP
# -------------------------
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")

# -------------------------
# LOAD DATA
# -------------------------
movies = pd.read_csv("movies_clean.csv")
sample_ratings = pd.read_csv("ratings_sample.csv")

# make sure IDs have same type
movies["movieId"] = movies["movieId"].astype(int)
sample_ratings["movieId"] = sample_ratings["movieId"].astype(int)

# -------------------------
# CONTENT-BASED DATA
# -------------------------
content_df = (
    movies[["movieId", "title", "genres"]]
    .drop_duplicates(subset=["movieId"])
    .reset_index(drop=True)
)

content_df["genres"] = content_df["genres"].fillna("").astype(str)

# -------------------------
# CONTENT-BASED MODEL
# -------------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(content_df["genres"])

indices = pd.Series(content_df.index, index=content_df["title"]).drop_duplicates()

def get_recommendations(title, top_n=10):
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[::-1][1:top_n + 1]

    return content_df[["title", "genres"]].iloc[sim_indices]

# -------------------------
# POPULARITY MODEL
# -------------------------
def popularity_recommend(top_n=10):
    popular_movies = (
        movies["title"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    popular_movies.columns = ["title", "number_of_ratings"]
    return popular_movies

# -------------------------
# COLLABORATIVE FILTERING MODEL
# -------------------------
movie_lookup = (
    movies[["movieId", "title"]]
    .drop_duplicates(subset=["movieId"], keep="first")
)

collab_df = sample_ratings.merge(
    movie_lookup,
    on="movieId",
    how="inner"
)

collab_df = collab_df.dropna(subset=["userId", "movieId", "title", "rating"])

# reduce size for Streamlit
movie_counts = collab_df["movieId"].value_counts()
user_counts = collab_df["userId"].value_counts()

collab_df = collab_df[
    collab_df["movieId"].isin(movie_counts[movie_counts > 1].index)
]

collab_df = collab_df[
    collab_df["userId"].isin(user_counts[user_counts > 1].index)
]

user_movie_matrix = collab_df.pivot_table(
    index="userId",
    columns="title",
    values="rating",
    fill_value=0
)

user_mean = user_movie_matrix.mean(axis=1)
user_movie_matrix_norm = user_movie_matrix.sub(user_mean, axis=0)

user_movie_matrix_norm = user_movie_matrix_norm.fillna(0)

def collaborative_recommend(title, top_n=10):
    if title not in user_movie_matrix_norm.columns:
        return pd.DataFrame()

    movie_vector = user_movie_matrix_norm[title].values.reshape(1, -1)

    similarities = cosine_similarity(
        movie_vector,
        user_movie_matrix_norm.T
    ).flatten()

    recs = pd.DataFrame({
        "title": user_movie_matrix_norm.columns,
        "similarity_score": similarities
    })

    recs = recs[recs["title"] != title]

    recs = recs.sort_values(
        by="similarity_score",
        ascending=False
    ).head(top_n)

    return recs

@st.cache_data
def fetch_poster(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": API_KEY,
        "query": movie_title.split("(")[0]  # cleaner search
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["results"]:
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    return None

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("🎬 Movie Recommendation System")

st.write(
    "This app recommends movies using popularity-based, content-based, "
    "and collaborative filtering recommendation techniques."
)

model_type = st.selectbox(
    "Choose recommendation model",
    [
        "Content-Based Recommendation",
        "Popularity-Based Recommendation",
        "Collaborative Filtering Recommendation"
    ]
)

if model_type == "Content-Based Recommendation":
    st.info("This model recommends movies with similar genres to the movie you choose.")

    movie = st.selectbox(
        "Choose a movie",
        sorted(content_df["title"].unique())
    )

    if st.button("Recommend"):
        recommendations = get_recommendations(movie)

        if recommendations.empty:
            st.error("Movie not found.")
        else:
            st.subheader("Recommended Movies")

            for i, row in recommendations.iterrows():
                poster_url = fetch_poster(row["title"])

                col1, col2 = st.columns([1, 3])

                with col1:
                    if poster_url:
                        st.image(poster_url)

                with col2:
                    st.write(f"🎬 {row['title']}")
                    st.write(f"🎭 {row['genres']}")

elif model_type == "Popularity-Based Recommendation":
    st.info("This model recommends the most popular movies based on the number of ratings.")

    if st.button("Show Popular Movies"):
        recommendations = popularity_recommend()

        st.subheader("Most Popular Movies")

        for i, row in recommendations.iterrows():
            poster_url = fetch_poster(row["title"])

    col1, col2 = st.columns([1, 3])

    with col1:
        if poster_url:
            st.image(poster_url)
        else:
            st.write("No image")

    with col2:
        st.write(f"🎬 {row['title']}")
        if "genres" in row:
            st.write(f"🎭 {row['genres']}")

else:
    st.info("This model recommends movies based on similar user rating patterns from the sample ratings data.")

    if user_movie_matrix.empty:
        st.error("Collaborative filtering data is empty after filtering. Reduce the filtering threshold.")
    else:
        movie = st.selectbox(
            "Choose a movie",
            sorted(user_movie_matrix.columns)
        )

        if st.button("Recommend"):
            recommendations = collaborative_recommend(movie)

            if recommendations.empty:
                st.error("No collaborative recommendations found.")
            else:
                st.subheader("Collaborative Filtering Recommendations")

                for i, row in recommendations.iterrows():
                    poster_url = fetch_poster(row["title"])

    col1, col2 = st.columns([1, 3])

    with col1:
        if poster_url:
            st.image(poster_url)
        else:
            st.write("No image")

    with col2:
        st.write(f"🎬 {row['title']}")
        if "genres" in row:
            st.write(f"🎭 {row['genres']}")