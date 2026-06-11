# Ranking Experiment Report

Primary objective: maximize Recall@10, with NDCG@10 as secondary tie-breaker.

```text
   phase                                                                 model  precision_at_10  recall_at_10  ndcg_at_10     rmse  n_users                                                                   notes  validation_recall_at_10
Baseline                                                         Legacy Hybrid           0.0515        0.0391      0.0656  0.89976      200                   Mean top-20 content profile + avg_rating*log1p(count)                      NaN
      P2                                      Weighted Content Profile (top20)           0.0515        0.0381      0.0660  0.89976      200                       profile=sum(rating*tfidf)/sum(rating), ratings>=4                   0.0336
      P2                                      Weighted Content Profile (top50)           0.0500        0.0365      0.0648  0.89976      200                       profile=sum(rating*tfidf)/sum(rating), ratings>=4                   0.0360
      P2                                  Weighted Content Profile (all_liked)           0.0490        0.0366      0.0642  0.89976      200                       profile=sum(rating*tfidf)/sum(rating), ratings>=4                   0.0357
      P4                                           Bayesian Popularity (m=100)           0.0365        0.0282      0.0454  0.89976      200                    Best m selected by validation Recall@10 then NDCG@10                      NaN
      P1         Tuned Hybrid {'cf': 0.5, 'cb': 0.4, 'pop': 0.1, 'genre': 0.0}           0.0370        0.0285      0.0424  0.89976      200                  Weights selected on validation Recall@10, then NDCG@10                      NaN
      P3 Tuned Hybrid + Genre {'cf': 0.5, 'cb': 0.4, 'pop': 0.1, 'genre': 0.0}           0.0370        0.0285      0.0424  0.89976      200                          Genre affinity added as a fourth hybrid signal                      NaN
   P5-P7                                                   XGBRanker rank:ndcg           0.0535        0.0348      0.0621      N/A      200 Binary relevance, 5x unseen negatives per positive, 11 ranking features                      NaN
```


## Best Models

- Best hybrid: Tuned Hybrid {'cf': 0.5, 'cb': 0.4, 'pop': 0.1, 'genre': 0.0}
- Best overall: Legacy Hybrid

## Interpretation

- RMSE is reported for SVD-backed rating prediction. XGBRanker emits relevance scores, not calibrated ratings, so RMSE is marked N/A for that model.
- Hybrid metrics move when candidate ordering changes; SVD RMSE can stay fixed because the CF submodel is unchanged.
- XGBRanker is production-relevant when serving can afford feature generation for a candidate set and when online ranking metrics matter more than rating prediction error.

## XGBRanker Feature Importance

```text
           feature       gain
            cb_sim 184.282959
movie_rating_count  60.750031
         pop_score  55.329243
           cf_pred  23.644123
   svd_dot_product  21.029526
  movie_avg_rating  11.987447
    genre_affinity   6.754106
 user_rating_count   5.861634
   user_avg_rating   4.143806
  item_factor_norm   3.689654
  user_factor_norm   2.479047
```
