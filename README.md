# 🎬 Personalised Movie Recommendation System Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-Recommendation_System-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---


   On Youtube every minute people upload 500 hours of videos, i.e. it would take 82 years to a user to watch all videos uploaded just in the last hour and on Spotify users can listen to ore than 80 million song tracks and podcasts so with the everchanging complexity of how we interact with films on streaming platforms and such platforms there has been a need to reduce the time a user spends while scrolling through content,as these platforms can have thousands of movies.It brings up an almost impossible move of getting to go through all the movies creating an instance known as **scrolling fatigue**.We are tasked with the creation of a recommender sytem which will help boost user interaction in a given film streamimg site,reduce scrolling fatigue and perhaps boost sales.This Project will also incorporate an active user interface for the target users before we reach the deployment stage.

  ## 🌟 Highlights

  - Create a popularity-based recommender system(baseline).
  - Create a content-based recommender system.
  - Use K-means and collaborative filtering to capture complex relationships.
  - Create a hybrid model which will be our final model.
  - Evaluation of the hybrid model.

## ℹ️ Overview

This project aims to use collaborative filtering,content-based recommendations among other tools to design a system which will recommend movies for a growing online movie streamimg platform.We will be working on this project as a group as each one is undertaking different parts of the project.

### ✍️ Authors

- Layla Abdikrim (ML & Modelling)
- Dave Ndung'u (ML & Modelling)
- Mading Garang (Presentations & Data Preparation)
- Trevor Obonyo (ML & Model Tuning)
- Clive Kinyanjui (Business Intelligence Tools)

## 📊 Dataset

- Source: *https://grouplens.org/datasets/movielens/latest/*
-  Key features:
  - User ID  
  - Item ID (Movie/Product/etc.)  
  - Ratings  
  - Additional metadata (genre, title, etc.)

## 🛠️ Tech Stack

-- Python 🐍  
- Pandas & NumPy  
- Scikit-learn  
- Surprise Library (SVD)  
- Matplotlib & Seaborn

## 🧠 Methodology

### 1. Data Preprocessing
- Handling missing values  
- Encoding users and items  
- Creating a user-item interaction matrix  

### 2. Exploratory Data Analysis (EDA)
- Rating distribution analysis  
- Most popular items  
- User engagement patterns  

### 3. K-Means Clustering
- Grouping similar users/items  
- Optional dimensionality reduction (PCA)  
- Cluster interpretation and insights  

### 4. SVD (Matrix Factorization)
- Decomposing user-item matrix into latent factors  
- Learning hidden relationships  
- Predicting missing ratings  

#### Example Usage


    def hybrid_recommendations(user_id, movies, ratings, best_model, genre_similarity, alpha = 0.5):

    # Getting CF (SVD) and CBF (genre_similarity) predictions for the user
    svd_df = get_svd_predictions(user_id, movies, ratings, best_model)
    genre_df = get_genre_scores(user_id, ratings, movies, genre_similarity)

    # merging both dataframes
    hybrid_df = svd_df.merge(genre_df, on = 'movieId')

    # Computing final score (weighted blend)
    hybrid_df['final_score'] = alpha * hybrid_df['svd_score'] + (1 - alpha) * hybrid_df['genre_score']

    # Get top recommendations
    top_movies = hybrid_df.sort_values(by = 'final_score', ascending = False).head(10)

    top_movies = top_movies.merge(movies[['movieId', 'title']], on = 'movieId')

    return top_movies[['movieId', 'title', 'final_score']]



### 5. Recommendation Generation
- Top-N recommendations per user  
- Optional hybrid approach (content-based recommender + SVD)

## 📈 Results
- Model evaluation metrics (RMSE / MAE)  
- Cluster insights and interpretations  
- Sample recommendations generated

## 🚀 How to Run the Project

```bash
# Clone repository
git clone https://github.com/Wanyiri9dave/similarityEngine.git
# Navigate to project
cd SimilarityEngine

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook
```
## 🔮 Future Improvements

- Implement deep learning-based recommender systems
- Deploy using SFlask
- Build real-time recommendation API
- Improve hybrid model performance


---

⭐ If you like this project, consider starring the repo!














  
