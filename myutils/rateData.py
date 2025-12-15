from myutils.query import get_movie_ratings, get_user_ratings

def get_average_rating(movie_id):
    ratings = get_movie_ratings(movie_id)
    if ratings:
        return sum(ratings) / len(ratings)
    return 0

def get_user_rating(user_id, movie_id):
    user_ratings = get_user_ratings(user_id)
    return user_ratings.get(movie_id, None)

def get_top_rated_movies(n=10):
    # This function should return the top n movies based on average ratings
    # Assuming we have a function to get all movie IDs
    all_movie_ids = get_all_movie_ids()
    movie_ratings = {movie_id: get_average_rating(movie_id) for movie_id in all_movie_ids}
    top_movies = sorted(movie_ratings.items(), key=lambda x: x[1], reverse=True)
    return top_movies[:n]