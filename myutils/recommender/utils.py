def load_movie_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def preprocess_movie_data(movie_data):
    # Example preprocessing steps
    movie_data['genres'] = movie_data['genres'].apply(lambda x: x.split(','))
    return movie_data

def calculate_similarity(movie_data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_data['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, cosine_sim, movie_data):
    indices = pd.Series(movie_data.index, index=movie_data['title']).drop_duplicates()
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 recommendations

    movie_indices = [i[0] for i in sim_scores]
    return movie_data.iloc[movie_indices]