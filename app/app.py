import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

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

def collaborative_recommend(title, top_n=10):
    if title not in user_movie_matrix.columns:
        return pd.DataFrame()

    selected_movie = user_movie_matrix[[title]].T
    similarities = cosine_similarity(selected_movie, user_movie_matrix.T).flatten()

    recs = pd.DataFrame({
        "title": user_movie_matrix.columns,
        "similarity_score": similarities
    })

    recs = (
        recs[recs["title"] != title]
        .sort_values(by="similarity_score", ascending=False)
        .head(top_n)
    )

    return recs

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
            st.dataframe(recommendations, use_container_width=True)

elif model_type == "Popularity-Based Recommendation":
    st.info("This model recommends the most popular movies based on the number of ratings.")

    if st.button("Show Popular Movies"):
        recommendations = popularity_recommend()

        st.subheader("Most Popular Movies")
        st.dataframe(recommendations, use_container_width=True)

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
                st.dataframe(recommendations, use_container_width=True)