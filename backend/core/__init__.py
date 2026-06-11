# Core recommendation engine modules
mmr_lambda: float = 0.7      # 70% relevance, 30% diversity
mmr_candidates: int = 50     # re-rank top 50 before picking final 10