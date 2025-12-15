from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

class ContentBasedRecommender:
    def __init__(self, movie_data):
        self.movie_data = movie_data
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movie_data['description'])

    def get_recommendations(self, title, top_n=10):
        # Compute the cosine similarity matrix
        cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        # Get the index of the movie that matches the title
        idx = self.movie_data.index[self.movie_data['title'] == title].tolist()[0]

        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the top_n most similar movies
        sim_scores = sim_scores[1:top_n + 1]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top_n most similar movies
        return self.movie_data.iloc[movie_indices]