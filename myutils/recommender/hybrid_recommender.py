from myutils.recommender.content_based import ContentBasedRecommender
from myutils.recommender.collaborative_filtering import CollaborativeFilteringRecommender

class HybridRecommender:
    def __init__(self, content_data, user_data):
        self.content_recommender = ContentBasedRecommender(content_data)
        self.collaborative_recommender = CollaborativeFilteringRecommender(user_data)

    def recommend(self, user_id, movie_id, n_recommendations=10):
        content_based_recommendations = self.content_recommender.recommend(movie_id, n_recommendations)
        collaborative_recommendations = self.collaborative_recommender.recommend(user_id, n_recommendations)

        # Combine recommendations
        combined_recommendations = self.combine_recommendations(content_based_recommendations, collaborative_recommendations)

        return combined_recommendations

    def combine_recommendations(self, content_recommendations, collaborative_recommendations):
        # Simple combination logic: prioritize collaborative recommendations
        recommendations = {movie: score for movie, score in collaborative_recommendations}
        for movie in content_recommendations:
            if movie not in recommendations:
                recommendations[movie] = content_recommendations[movie]

        # Sort recommendations by score
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

        return sorted_recommendations[:10]  # Return top 10 recommendations