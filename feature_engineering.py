"""Canonical placement features shared by training and Streamlit prediction."""

import pandas as pd


BASE_FEATURES = [
    "gender", "age", "degree", "branch", "cgpa", "backlogs", "internships",
    "certifications", "coding_skills", "communication_skills", "aptitude_score", "projects",
]
ENGINEERED_FEATURES = ["overall_skill_score", "academic_performance_index", "profile_strength"]
MODEL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES


def add_placement_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact feature engineering from Notebook 04 without changing its input."""
    required = set(BASE_FEATURES)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing feature-engineering inputs: {sorted(missing)}")
    result = data.copy()
    result["overall_skill_score"] = (
        result["coding_skills"] + result["communication_skills"] + result["aptitude_score"]
    ) / 3
    result["academic_performance_index"] = result["cgpa"] * 10 - result["backlogs"] * 5
    result["profile_strength"] = (
        result["internships"] + result["projects"] + result["certifications"]
    )
    return result
