import numpy as np
import pandas as pd

def recall_at_k(recommended_items, relevant_items):
    """
    Computes Recall@k for a single user.
    
    Args:
        recommended_items (dict): A mapping of userId to movieId recommendations.
        relevant_items (set): The set of movies from the test split that the user liked.
    
    Returns:
        float: Recall@k value for the user.
    """

    if not relevant_items:
        return None
    hits = len(set(recommended_items) & set(relevant_items))
    return hits / len(relevant_items)

def average_recall_at_k(recommendations, ground_truth):
    """
    Computes the average Recall@k for all of the passed recommendations.
    
    Args:
        recommendations (dict): A mapping of userIds to movieId recommendations.
        ground_truth (dict): A mapping of userIds to withheld movieId recommendations.
    
    Returns:
        The average Recall@k value over all of the passed recommendations.
    """
    recalls = []

    for user_id in recommendations:
        recs = recommendations[user_id]
        true_items = ground_truth.get(user_id, set())
        score = recall_at_k(recs, true_items)
        if score is not None:
            recalls.append(score)

    return np.mean(recalls) if recalls else 0.0