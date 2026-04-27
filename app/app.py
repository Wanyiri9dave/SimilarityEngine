import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import numpy as np

API_KEY = "022b877171805be41e9b2ffd24e5ded3"

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def clamp_score(score):
    """Clamp similarity score to [0, 1] range for progress bar display."""
    return max(0.0, min(1.0, float(score)))

# -------------------------
# PAGE SETUP
# -------------------------
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies_clean.csv")
    sample_ratings = pd.read_csv("ratings_sample.csv")
    movies["movieId"] = movies["movieId"].astype(int)
    sample_ratings["movieId"] = sample_ratings["movieId"].astype(int)
    return movies, sample_ratings

movies, sample_ratings = load_data()

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
    sim_scores_sorted = sim_scores[sim_indices]

    recs = content_df[["title", "genres"]].iloc[sim_indices].copy()
    # Normalize content scores: divide by max to get realistic percentages
    max_score = sim_scores_sorted.max()
    if max_score > 0:
        recs["similarity_score"] = sim_scores_sorted / max_score
    else:
        recs["similarity_score"] = sim_scores_sorted
    return recs

# -------------------------
# POPULARITY MODEL
# -------------------------
@st.cache_data
def popularity_recommend(top_n=10):
    popular_movies = (
        movies["title"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    popular_movies.columns = ["title", "number_of_ratings"]
    # Scale to realistic 0-1 range
    min_ratings = popular_movies["number_of_ratings"].min()
    max_ratings = popular_movies["number_of_ratings"].max()
    
    if max_ratings == min_ratings:
        popular_movies["similarity_score"] = 0.5
    else:
        popular_movies["similarity_score"] = (popular_movies["number_of_ratings"] - min_ratings) / (max_ratings - min_ratings)
    
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

# Better filtering for collaborative filtering
movie_counts = collab_df["movieId"].value_counts()
user_counts = collab_df["userId"].value_counts()

# Keep movies with at least 1 rating and users with at least 1 rating (more lenient)
collab_df = collab_df[
    collab_df["movieId"].isin(movie_counts[movie_counts >= 1].index)
]

collab_df = collab_df[
    collab_df["userId"].isin(user_counts[user_counts >= 1].index)
]

# Create user-movie matrix with normalized ratings
user_movie_matrix = collab_df.pivot_table(
    index="userId",
    columns="title",
    values="rating",
    fill_value=0
)

# Initialize movie_similarity_df as empty
movie_similarity_df = pd.DataFrame()

# Create movie-movie similarity matrix based on user preferences only if data exists
if not user_movie_matrix.empty and len(user_movie_matrix.columns) > 1:
    # Normalize the matrix to handle rating scale differences
    movie_matrix_normalized = user_movie_matrix.fillna(0).astype(float)
    
    # Standardize each movie's ratings (handle case where std = 0)
    movie_mean = movie_matrix_normalized.mean()
    movie_std = movie_matrix_normalized.std()
    movie_std[movie_std == 0] = 1  # Avoid division by zero
    
    movie_matrix_normalized = (movie_matrix_normalized - movie_mean) / movie_std
    movie_matrix_normalized = movie_matrix_normalized.fillna(0)

    # Calculate item-item similarity
    if movie_matrix_normalized.shape[1] > 1:  # Need at least 2 movies
        movie_similarity_matrix = cosine_similarity(movie_matrix_normalized.T)
        movie_similarity_df = pd.DataFrame(
            movie_similarity_matrix,
            index=user_movie_matrix.columns,
            columns=user_movie_matrix.columns
        )

@st.cache_data
def collaborative_recommend(title, top_n=10):
    """
    Item-based collaborative filtering recommendation.
    Finds movies similar to the selected movie based on user rating patterns.
    """
    if movie_similarity_df.empty or title not in movie_similarity_df.columns:
        return pd.DataFrame()

    # Get similarity scores for the selected movie
    sim_scores = movie_similarity_df[title]
    
    # Exclude the movie itself
    similar_movies = sim_scores[sim_scores.index != title]
    
    # Filter by minimum threshold and sort
    similar_movies = similar_movies.sort_values(ascending=False).head(top_n)

    if similar_movies.empty:
        return pd.DataFrame()

    # Min-max normalization to [0, 1] range based on actual data range
    min_score = similar_movies.min()
    max_score = similar_movies.max()
    
    if max_score == min_score:
        # All scores are the same, use uniform distribution
        normalized_scores = np.full(len(similar_movies), 0.5)
    else:
        # Scale to [0, 1] range
        normalized_scores = (similar_movies.values - min_score) / (max_score - min_score)

    recs = pd.DataFrame({
        "title": similar_movies.index,
        "similarity_score": normalized_scores
    }).reset_index(drop=True)

    return recs

def hybrid_recommend(title, top_n=10, content_weight=0.5, collab_weight=0.5):
    """
    Hybrid recommendation combining content-based and collaborative filtering.
    """
    recs_combined = pd.DataFrame()
    
    # Get content-based recommendations with preserved scores
    if title in indices:
        content_recs = get_recommendations(title, top_n=top_n)
        if not content_recs.empty:
            content_recs = content_recs.copy()
            content_recs["source"] = "content"
            content_recs["weighted_score"] = content_recs["similarity_score"] * content_weight
            recs_combined = pd.concat([recs_combined, content_recs], ignore_index=True)
    
    # Get collaborative filtering recommendations
    if title in movie_similarity_df.columns and not movie_similarity_df.empty:
        collab_recs = collaborative_recommend(title, top_n=top_n)
        if not collab_recs.empty:
            collab_recs = collab_recs.copy()
            collab_recs["source"] = "collaborative"
            collab_recs["weighted_score"] = collab_recs["similarity_score"] * collab_weight
            recs_combined = pd.concat([recs_combined, collab_recs], ignore_index=True)
    
    if recs_combined.empty:
        return pd.DataFrame()
    
    # Aggregate scores from both methods (sum weighted scores)
    recs_agg = recs_combined.groupby("title").agg({
        "weighted_score": "sum"
    }).sort_values("weighted_score", ascending=False).head(top_n)
    
    # Normalize hybrid scores to [0, 1] range based on actual data range
    min_score = recs_agg["weighted_score"].min()
    max_score = recs_agg["weighted_score"].max()
    
    if max_score == min_score:
        normalized_scores = np.full(len(recs_agg), 0.5)
    else:
        normalized_scores = (recs_agg["weighted_score"].values - min_score) / (max_score - min_score)
    
    return pd.DataFrame({
        "title": recs_agg.index,
        "similarity_score": normalized_scores
    }).reset_index(drop=True)

def get_movie_info(title):
    """Get additional info about a movie."""
    movie_info = movies[movies["title"] == title]
    if not movie_info.empty:
        row = movie_info.iloc[0]
        return {
            "title": row.get("title", "N/A"),
            "genres": row.get("genres", "N/A"),
            "year": row.get("year", "N/A"),
            "avg_rating": row.get("avg_rating", "N/A")
        }
    return None

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
    "This app recommends movies using content-based, popularity-based, "
    "collaborative filtering, and hybrid recommendation techniques."
)

# Sidebar
with st.sidebar:
    st.header("📊 Dataset Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Movies", len(movies))
        st.metric("Movies in Content DB", len(content_df))
    with col2:
        st.metric("Total Ratings", len(sample_ratings))
        if not user_movie_matrix.empty:
            st.metric("Unique Users", user_movie_matrix.shape[0])
    
    st.divider()
    
    st.header("⚙️ Settings")
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    
    st.divider()
    
    model_type = st.radio(
        "Choose recommendation model",
        [
            "🎭 Content-Based",
            "⭐ Popularity-Based",
            "👥 Collaborative Filtering",
            "🔀 Hybrid (Combined)"
        ]
    )

# Search box
st.subheader("🔍 Find a Movie")
search_query = st.text_input("Search for a movie...", placeholder="e.g., The Matrix, Inception")

if search_query:
    matching_movies = content_df[content_df["title"].str.contains(search_query, case=False, na=False)]["title"].unique()
    if len(matching_movies) > 0:
        st.write(f"Found {len(matching_movies)} matching movie(s)")
        selected_movie = st.selectbox("Select a movie:", matching_movies, key="search_select")
    else:
        st.warning("No movies found matching your search.")
        selected_movie = None
else:
    selected_movie = None

st.divider()

# Recommendations based on model type
if model_type == "🎭 Content-Based":
    st.info("📌 This model recommends movies with similar genres to the movie you choose.")
    
    if not selected_movie:
        movie = st.selectbox(
            "Choose a movie",
            sorted(content_df["title"].unique()),
            key="content_select"
        )
    else:
        movie = selected_movie

    if st.button("Get Recommendations", key="content_btn"):
        recommendations = get_recommendations(movie, top_n=top_n)

        if recommendations.empty:
            st.error("❌ Movie not found or no recommendations available.")
        else:
            st.subheader(f"🎬 Recommended Movies (Similar to: {movie})")
            
            # Display in grid layout
            cols = st.columns(3)
            for idx, (i, row) in enumerate(recommendations.iterrows()):
                col = cols[idx % 3]
                with col:
                    poster_url = fetch_poster(row["title"])
                    
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.write("🎬 No image available")
                    
                    st.write(f"**{row['title']}**")
                    st.caption(f"🎭 {row['genres']}")
                    st.progress(clamp_score(row["similarity_score"]))

elif model_type == "⭐ Popularity-Based":
    st.info("📌 This model recommends the most popular movies based on the number of ratings.")

    if st.button("Show Popular Movies", key="pop_btn"):
        recommendations = popularity_recommend(top_n=top_n)

        st.subheader("⭐ Most Popular Movies")
        
        # Display in grid layout
        cols = st.columns(3)
        for idx, (i, row) in enumerate(recommendations.iterrows()):
            col = cols[idx % 3]
            with col:
                poster_url = fetch_poster(row["title"])
                
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.write("🎬 No image available")
                
                st.write(f"**{row['title']}**")
                st.caption(f"👁️ {int(row['number_of_ratings'])} ratings")
                st.progress(clamp_score(row["similarity_score"]))

elif model_type == "👥 Collaborative Filtering":
    st.info("📌 This model recommends movies based on similar user rating patterns.")

    if user_movie_matrix.empty or movie_similarity_df.empty:
        st.error("⚠️ Insufficient collaborative filtering data.")
        st.write(f"**Debug Info:** Movies in dataset: {len(content_df)}, Movies in collaborative data: {len(user_movie_matrix.columns) if not user_movie_matrix.empty else 0}")
    else:
        st.caption(f"📊 Collaborative Data: {len(user_movie_matrix)} users, {len(user_movie_matrix.columns)} movies")
        
        if not selected_movie:
            movie = st.selectbox(
                "Choose a movie",
                sorted(movie_similarity_df.columns),
                key="collab_select"
            )
        else:
            movie = selected_movie

        if st.button("Get Recommendations", key="collab_btn"):
            with st.spinner("⏳ Generating recommendations..."):
                recommendations = collaborative_recommend(movie, top_n=top_n)

            if recommendations.empty:
                st.warning("⚠️ No similar movies found based on user ratings. Try another movie!")
            else:
                st.subheader(f"👥 Collaborative Recommendations (Similar to: {movie})")
                
                # Display in grid layout
                cols = st.columns(3)
                for idx, (i, row) in enumerate(recommendations.iterrows()):
                    col = cols[idx % 3]
                    with col:
                        poster_url = fetch_poster(row["title"])
                        
                        if poster_url:
                            st.image(poster_url, use_container_width=True)
                        else:
                            st.write("🎬 No image available")
                        
                        st.write(f"**{row['title']}**")
                        st.progress(clamp_score(row["similarity_score"]))

else:  # Hybrid
    st.info("📌 This model combines content-based and collaborative filtering for best results.")

    if not selected_movie:
        all_movies = sorted(set(list(content_df["title"].unique()) + list(movie_similarity_df.columns if not movie_similarity_df.empty else [])))
        movie = st.selectbox(
            "Choose a movie",
            all_movies,
            key="hybrid_select"
        )
    else:
        movie = selected_movie

    if st.button("Get Recommendations", key="hybrid_btn"):
        with st.spinner("⏳ Generating recommendations..."):
            recommendations = hybrid_recommend(movie, top_n=top_n)

        if recommendations.empty:
            st.error("❌ No recommendations available for this movie.")
        else:
            st.subheader(f"🔀 Hybrid Recommendations (Similar to: {movie})")
            
            # Display in grid layout
            cols = st.columns(3)
            for idx, (i, row) in enumerate(recommendations.iterrows()):
                col = cols[idx % 3]
                with col:
                    poster_url = fetch_poster(row["title"])
                    
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.write("🎬 No image available")
                    
                    st.write(f"**{row['title']}**")
                    st.progress(clamp_score(row["similarity_score"]))