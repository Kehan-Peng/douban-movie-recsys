from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class CollaborativeFiltering:
    def __init__(self, ratings_data):
        self.ratings_data = ratings_data
        self.user_similarity = None
        self.item_similarity = None

    def compute_user_similarity(self):
        user_item_matrix = self.ratings_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        self.user_similarity = cosine_similarity(user_item_matrix)
        return self.user_similarity

    def compute_item_similarity(self):
        item_user_matrix = self.ratings_data.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
        self.item_similarity = cosine_similarity(item_user_matrix)
        return self.item_similarity

    def get_user_recommendations(self, user_id, num_recommendations=5):
        user_item_matrix = self.ratings_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        user_index = user_item_matrix.index.get_loc(user_id)
        similar_users = list(enumerate(self.user_similarity[user_index]))
        similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:]  # Exclude self

        recommendations = {}
        for similar_user_index, similarity_score in similar_users:
            similar_user_id = user_item_matrix.index[similar_user_index]
            similar_user_ratings = user_item_matrix.loc[similar_user_id]
            for movie_id, rating in similar_user_ratings.items():
                if rating > 0 and movie_id not in recommendations:
                    recommendations[movie_id] = similarity_score * rating

        recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        return [movie[0] for movie in recommended_movies]

    def get_item_recommendations(self, movie_id, num_recommendations=5):
        item_index = self.ratings_data['movie_id'].unique().tolist().index(movie_id)
        similar_items = list(enumerate(self.item_similarity[item_index]))
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:]  # Exclude self

        recommended_movies = [self.ratings_data['movie_id'].unique()[i[0]] for i in similar_items[:num_recommendations]]
        return recommended_movies
