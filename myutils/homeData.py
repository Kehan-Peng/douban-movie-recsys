from myutils.recommender.hybrid_recommender import HybridRecommender

def get_home_data(user_id):
    # Initialize the hybrid recommender
    recommender = HybridRecommender()

    # Get recommended movies for the user
    recommended_movies = recommender.get_recommendations(user_id)

    # Prepare home data
    home_data = {
        'recommended_movies': recommended_movies,
        # Add any other home page data you want to include
    }

    return home_data