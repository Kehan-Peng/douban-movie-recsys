from __future__ import annotations

from typing import Dict, List, Optional

from myutils.recommend import get_algorithm_recommendations, recommend_movies, recommend_similar_movies


class RecommendationService:
    def recommend_for_user(self, user_email: Optional[str], top_n: int = 10) -> List[Dict]:
        return recommend_movies(user_email, top_n)

    def recommend_similar(self, movie_id: int, top_n: int = 6) -> List[Dict]:
        return recommend_similar_movies(movie_id, top_n)

    def recommend_with_algorithm(self, algorithm: str, user_email: Optional[str], top_n: int = 10) -> List[Dict]:
        return get_algorithm_recommendations(algorithm, user_email, top_n)
