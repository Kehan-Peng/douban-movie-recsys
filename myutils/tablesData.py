from myutils.query import get_movie_data, get_user_ratings

def get_movie_tables():
    """
    Fetches movie data from the database and organizes it into tables.
    
    Returns:
        dict: A dictionary containing movie data organized by various attributes.
    """
    movie_data = get_movie_data()
    tables = {
        'titles': [movie['title'] for movie in movie_data],
        'ratings': [movie['rating'] for movie in movie_data],
        'genres': [movie['genre'] for movie in movie_data],
        'directors': [movie['director'] for movie in movie_data],
        'actors': [movie['actors'] for movie in movie_data],
    }
    return tables

def get_user_movie_ratings(user_id):
    """
    Fetches movie ratings for a specific user.
    
    Args:
        user_id (int): The ID of the user for whom to fetch ratings.
        
    Returns:
        list: A list of movie ratings given by the user.
    """
    return get_user_ratings(user_id)