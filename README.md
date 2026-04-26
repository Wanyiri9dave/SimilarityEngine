# 🎬 Personalised Movie Recommendation System Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-Recommendation_System-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

 
## 📌 Introduction

On platforms like YouTube, over 500 hours of video are uploaded every minute, meaning it would take a user approximately 82 years to watch all content uploaded in just one hour. Similarly, Spotify offers access to more than 80 million songs and podcasts.

With the rapid growth of digital content, users often struggle to find relevant movies on streaming platforms. This leads to a phenomenon known as scrolling fatigue, where users spend excessive time searching instead of consuming content.

This project aims to design a Movie Recommendation System that improves user experience by reducing search time, increasing engagement, and enhancing content discovery. The system also includes an interactive user interface built using Streamlit.

## 🌟 Highlights
Developed a Popularity-Based Recommendation System (baseline model)
Built a Content-Based Recommendation System
Implemented Collaborative Filtering (SVD-based)
Designed a Hybrid Recommendation Model
Created an interactive Streamlit Web Application
Evaluated model performance using appropriate metrics
## ℹ️ Overview

This project combines multiple recommendation techniques to build a robust system for a movie streaming platform. It leverages:

User behavior (ratings)
Movie metadata (genres, titles)
Machine learning models

The system supports multiple recommendation approaches and allows users to interact with the models through a user-friendly interface.

## ✍️ Authors
Leila Abdikarim – Machine Learning & Modelling  
Dave Ndung'u – Machine Learning & Modelling  
Mading Garang – Presentations & Data Preparation  
Trevor Obonyo – Model Tuning  
Clive Kinyanjui – Business Intelligence Tools

## 📊 Dataset
Source: https://grouplens.org/datasets/movielens/latest/
Key Features:
User ID
Movie ID
Ratings
Movie metadata (title, genres, etc.)

## 🛠️ Tech Stack  
Python 🐍  
Pandas & NumPy  
Scikit-learn  
Surprise Library (SVD)  
Matplotlib & Seaborn  
Streamlit

## 🧠 Methodology
1. Data Preprocessing
Handling missing values
Merging datasets
Creating a user-item interaction matrix
2. Exploratory Data Analysis (EDA)
Rating distribution analysis
Identifying most popular movies
Understanding user behavior patterns
3. Content-Based Filtering
TF-IDF vectorization of movie genres
Cosine similarity to identify similar movies
4. Collaborative Filtering (SVD)
Matrix factorization using Singular Value Decomposition
Learning latent features representing user preferences and movie characteristics
Predicting unseen ratings
## 5. Hybrid Recommendation Model

A combination of collaborative filtering and content-based methods:  

def hybrid_recommendations(user_id, movies, ratings, best_model, genre_similarity, alpha=0.5):  

# Collaborative filtering predictions  
    svd_df = get_svd_predictions(user_id, movies, ratings, best_model)  
    
 # Content-based scores  
      genre_df = get_genre_scores(user_id, ratings, movies, genre_similarity)  
      
  # Merge both  
     hybrid_df = svd_df.merge(genre_df, on='movieId')  
  # Weighted combination  
     hybrid_df['final_score'] = alpha * hybrid_df['svd_score'] + (1 - alpha) * hybrid_df['genre_score']  
     
   # Top recommendations  
      top_movies = hybrid_df.sort_values(by='final_score', ascending=False).head(10)

    top_movies = top_movies.merge(movies[['movieId', 'title']], on='movieId')

    return top_movies[['movieId', 'title', 'final_score']]


## 6. Recommendation Generation
Top-N recommendations
Model comparison (Content vs Collaborative vs Hybrid)
## 📱 Streamlit Application

The project includes an interactive Streamlit app where users can:

Select a movie
Choose a recommendation model:
Content-Based
Popularity-Based
Collaborative Filtering
View recommended movies instantly
## ▶️ Run the app:

```
streamlit run app.py
```
## 📈 Results
Generated accurate Top-N recommendations  
Improved recommendation quality using hybrid model  
Reduced user search effort  
Demonstrated model effectiveness through evaluation metrics (RMSE / MAE)
## 🎯 Business Value

This system helps streaming platforms to:  

Reduce user scrolling time  
Increase user engagement  
Improve content discoverability  
Enhance user satisfaction and retention

## 🔮 Future Improvements
Implement deep learning-based recommendation systems  
Deploy using Flask or cloud platforms  
Build real-time recommendation API  
Improve hybrid model performance  
Add movie posters and UI enhancements  


⭐ Support

If you found this project useful, consider starring the repository ⭐

















  
