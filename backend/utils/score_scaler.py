"""
Score Scaler Module
Convert raw model probability to a 300-900 credit score.
"""


def scale_probability_to_score(
    probability: float,
    min_score: int = 300,
    max_score: int = 900,
) -> int:
    """
    Scale a 0-1 default probability to a 300-900 credit score.

    Uses inverse mapping: lower probability of default = higher credit score.
    Formula: score = max_score - probability * (max_score - min_score)

    Args:
        probability: Model's predicted probability of default (0 to 1).
        min_score: Minimum credit score (default 300).
        max_score: Maximum credit score (default 900).

    Returns:
        Integer credit score between min_score and max_score.
    """
    raise NotImplementedError("To be implemented")


def score_to_risk_level(score: int) -> str:
    """
    Map a credit score to a risk level category.

    Risk levels:
        - LOW: score >= 700
        - MEDIUM: 500 <= score < 700
        - HIGH: score < 500

    Args:
        score: Credit score on 300-900 scale.

    Returns:
        Risk level string: 'LOW', 'MEDIUM', or 'HIGH'.
    """
    raise NotImplementedError("To be implemented")
