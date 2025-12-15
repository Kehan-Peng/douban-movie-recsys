from myutils.query import get_movie_data

def get_movie_types():
    # Fetch movie data from the database
    movie_data = get_movie_data()
    
    # Extract unique movie types/genres
    movie_types = set(movie['type'] for movie in movie_data)
    
    return list(movie_types)