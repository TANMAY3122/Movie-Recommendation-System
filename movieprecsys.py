# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Collaborative Filtering Part

# 1. Create a user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# 2. Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# 3. Define function for collaborative filtering recommendations
def get_user_recommendations(user_id, user_item_matrix, user_similarity, num_recommendations=5):
    user_idx = user_id - 1
    sim_scores = user_similarity[user_idx]
    weighted_ratings = (sim_scores[:, None] * user_item_matrix.fillna(0)).sum(axis=0)
    rated_by_similar_users = user_item_matrix.notna().sum(axis=0)
    recommendations = (weighted_ratings / rated_by_similar_users).sort_values(ascending=False).dropna()
    return recommendations.head(num_recommendations)

# Content-Based Filtering Part

# 1. Preprocess the genres into a bag of words format
movies['genres'] = movies['genres'].str.split('|')
movies['genre_str'] = movies['genres'].apply(lambda x: ' '.join(x))

# 2. Convert genres to a matrix using CountVectorizer
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies['genre_str'])

# 3. Compute cosine similarity for movies based on genres
movie_similarity = cosine_similarity(genre_matrix)

# 4. Define function for content-based recommendations
def get_content_recommendations(movie_title, movies, movie_similarity, num_recommendations=5):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(movie_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    similar_movies_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]
    return movies['title'].iloc[similar_movies_indices]

# Model Evaluation

# 1. Split ratings data for evaluation
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# 2. Define prediction function for collaborative filtering evaluation
def predict_rating(user_id, movie_id, user_item_matrix, user_similarity):
    user_idx = user_id - 1
    movie_idx = movie_id - 1
    sim_scores = user_similarity[user_idx]
    movie_ratings = user_item_matrix.iloc[:, movie_idx]
    rated_users = movie_ratings.dropna().index
    numerator = sum(sim_scores[ru] * movie_ratings[ru] for ru in rated_users)
    denominator = sum(abs(sim_scores[ru]) for ru in rated_users)
    return numerator / denominator if denominator != 0 else np.nan

# 3. Calculate RMSE for collaborative filtering
actual_ratings = []
predicted_ratings = []

for _, row in test_data.iterrows():
    user_id = int(row['userId'])
    movie_id = int(row['movieId'])
    actual_rating = row['rating']
    predicted_rating = predict_rating(user_id, movie_id, user_item_matrix, user_similarity)
    if not np.isnan(predicted_rating):
        actual_ratings.append(actual_rating)
        predicted_ratings.append(predicted_rating)

rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
print("Root Mean Square Error (RMSE) for Collaborative Filtering:", rmse)

# Hybrid Recommendation System

# 1. Define function for hybrid recommendations
def get_hybrid_recommendations(user_id, movie_title, user_item_matrix, user_similarity, movie_similarity, movies, num_recommendations=5):
    cf_recommendations = get_user_recommendations(user_id, user_item_matrix, user_similarity, num_recommendations)
    cb_recommendations = get_content_recommendations(movie_title, movies, movie_similarity, num_recommendations)
    hybrid_recommendations = pd.concat([cf_recommendations, cb_recommendations]).drop_duplicates().head(num_recommendations)
    return hybrid_recommendations

# 2. Test hybrid recommendation system
user_id = 1
movie_title = "Toy Story (1995)"
hybrid_recs = get_hybrid_recommendations(user_id, movie_title, user_item_matrix, user_similarity, movie_similarity, movies)
print("\nHybrid Recommendations:")
print(hybrid_recs)
