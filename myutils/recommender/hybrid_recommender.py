from myutils.recommender.content_based import ContentBasedRecommender
from myutils.recommender.collaborative_filtering import CollaborativeFilteringRecommender

class HybridRecommender:
    def __init__(self, content_data, user_data):
        self.content_recommender = ContentBasedRecommender(content_data)
        self.collaborative_recommender = CollaborativeFilteringRecommender(user_data)
        self.movie_data = content_data

    def recommend(self, user_id, movie_id, n_recommendations=10):
        # 1. 获取内容推荐结果 [电影ID列表]
        content_recs = self.content_recommender.get_recommendations_by_id(movie_id, top_n=20)
        
        # 2. 获取协同过滤推荐结果 [(电影ID, 分数)]
        collab_recs = self.collaborative_recommender.get_user_recommendations(user_id, num_recommendations=20)

        # 3. 加权混合
        combined = self.hybrid_score(collab_recs, content_recs, weight_cf=0.6, weight_content=0.4)
        
        return combined[:n_recommendations]

    def hybrid_score(self, collab_recs, content_recs_with_score, weight_cf=0.6, weight_content=0.4):
        score_map = {}

        # 1. 归一化协同过滤分数
        cf_scores = [s for _, s in collab_recs]
        min_cf, max_cf = min(cf_scores), max(cf_scores)
        # 2. 归一化内容相似度
        cont_scores = [s for _, s in content_recs_with_score]
        min_cont, max_cont = min(cont_scores), max(cont_scores)

        # 录入CF归一化分数
        for mid, s in collab_recs:
            norm_cf = (s - min_cf) / (max_cf - min_cf) if max_cf != min_cf else 0
            score_map[mid] = norm_cf * weight_cf

        # 叠加内容相似度分数（带强弱区分）
        for mid, s in content_recs_with_score:
            norm_cont = (s - min_cont) / (max_cont - min_cont) if max_cont != min_cont else 0
            add_score = norm_cont * weight_content
            if mid in score_map:
                score_map[mid] += add_score
            else:
                # 小众/新电影：纯靠内容相似度排名
                score_map[mid] = add_score

        return sorted(score_map.items(), key=lambda x:x[1], reverse=True)