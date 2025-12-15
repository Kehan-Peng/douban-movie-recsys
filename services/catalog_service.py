from __future__ import annotations

from typing import Dict, List, Optional

from repositories.movie_repository import MovieRepository


class CatalogService:
    def __init__(self) -> None:
        self.repository = MovieRepository()

    def top_movies(self, limit: int = 10) -> List[Dict]:
        return self.repository.get_top_movies(limit)

    def search_movies(self, keyword: str) -> List[Dict]:
        return self.repository.search(keyword)

    def movie_detail(self, movie_id: int) -> Optional[Dict]:
        return self.repository.get_movie(movie_id)

    def movie_comments(self, movie_id: int) -> List[Dict]:
        return self.repository.get_comments(movie_id)
