"""
src — MovieMatcher source package.

Modules:
    data_loader    : load + join Kaggle CSVs, build content strings
    train          : MatrixFactorization (SGD) and BPR models
    recommend      : vectorized CF recommendation
    content_based  : TF-IDF similarity + hybrid CF+CB scoring
    evaluate       : RMSE, MAE, Precision@K, Recall@K, Coverage, Diversity
"""