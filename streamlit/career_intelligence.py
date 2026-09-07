"""Transparent, rule-based career guidance used by the Streamlit interface."""

from pathlib import Path
import pandas as pd

ROLE_TARGETS = {
    "Data Analyst": {"coding_skills": 6, "communication_skills": 7, "aptitude_score": 70, "cgpa": 7.0, "projects": 2, "internships": 1, "certifications": 1},
    "Data Scientist": {"coding_skills": 8, "communication_skills": 6, "aptitude_score": 78, "cgpa": 7.5, "projects": 3, "internships": 1, "certifications": 2},
    "ML Engineer": {"coding_skills": 8, "communication_skills": 6, "aptitude_score": 78, "cgpa": 7.5, "projects": 3, "internships": 2, "certifications": 2},
    "AI Engineer": {"coding_skills": 8, "communication_skills": 6, "aptitude_score": 75, "cgpa": 7.3, "projects": 3, "internships": 1, "certifications": 2},
    "NLP Engineer": {"coding_skills": 8, "communication_skills": 6, "aptitude_score": 75, "cgpa": 7.3, "projects": 3, "internships": 1, "certifications": 2},
    "Computer Vision Engineer": {"coding_skills": 8, "communication_skills": 6, "aptitude_score": 76, "cgpa": 7.4, "projects": 3, "internships": 1, "certifications": 2},
    "Deep Learning Engineer": {"coding_skills": 9, "communication_skills": 6, "aptitude_score": 80, "cgpa": 7.7, "projects": 3, "internships": 2, "certifications": 2},
    "Research Scientist": {"coding_skills": 8, "communication_skills": 7, "aptitude_score": 82, "cgpa": 8.0, "projects": 4, "internships": 1, "certifications": 2},
}
ROLE_SKILLS = {
    "Data Analyst": {"Excel", "SQL", "Power BI", "Data Visualization", "Python", "Statistics"},
    "Data Scientist": {"Python", "SQL", "Machine Learning", "Statistics", "Pandas", "TensorFlow"},
    "ML Engineer": {"Python", "Machine Learning", "MLOps", "PyTorch", "TensorFlow", "Docker"},
    "AI Engineer": {"Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "MLOps"},
    "NLP Engineer": {"Python", "NLP", "Deep Learning", "PyTorch", "Transformers", "TensorFlow"},
    "Computer Vision Engineer": {"Python", "Computer Vision", "Deep Learning", "OpenCV", "PyTorch", "TensorFlow"},
    "Deep Learning Engineer": {"Python", "Deep Learning", "PyTorch", "TensorFlow", "CUDA", "MLOps"},
    "Research Scientist": {"Python", "Statistics", "Machine Learning", "Deep Learning", "Research", "Pandas"},
}
WEIGHTS = {"coding_skills": .24, "communication_skills": .14, "aptitude_score": .20, "cgpa": .15, "projects": .11, "internships": .08, "certifications": .08}
DISPLAY_NAMES = {"coding_skills": "Coding", "communication_skills": "Communication", "aptitude_score": "Aptitude", "cgpa": "CGPA", "projects": "Projects", "internships": "Internships", "certifications": "Certifications"}


def role_matches(profile: dict) -> pd.DataFrame:
    scores = []
    selected_skills = set(profile.get("technical_skills", []))
    for role, targets in ROLE_TARGETS.items():
        profile_score = sum(WEIGHTS[key] * min(float(profile[key]) / value, 1) for key, value in targets.items())
        matched_skills = selected_skills.intersection(ROLE_SKILLS[role])
        # Explicit technical skills carry 40% of the role-match score when supplied.
        # Without a skill selection, preserve the earlier profile-only behaviour.
        score = profile_score if not selected_skills else (0.60 * profile_score + 0.40 * (len(matched_skills) / len(ROLE_SKILLS[role])))
        scores.append({"Role": role, "Profile match": round(score * 100, 1), "Matched skills": ", ".join(sorted(matched_skills)) or "No direct role skills selected"})
    return pd.DataFrame(scores).sort_values("Profile match", ascending=False).reset_index(drop=True)


def readiness_score(profile: dict) -> float:
    values = {"coding_skills": profile["coding_skills"] / 10, "communication_skills": profile["communication_skills"] / 10, "aptitude_score": profile["aptitude_score"] / 100, "cgpa": profile["cgpa"] / 10, "projects": min(profile["projects"] / 4, 1), "internships": min(profile["internships"] / 2, 1), "certifications": min(profile["certifications"] / 3, 1)}
    return round(sum(WEIGHTS[key] * value for key, value in values.items()) * 100, 1)


def skill_gaps(profile: dict, role: str) -> pd.DataFrame:
    rows = [{"Area": DISPLAY_NAMES[key], "Current": float(profile[key]), "Recommended": value, "Gap": round(max(value - float(profile[key]), 0), 1)} for key, value in ROLE_TARGETS[role].items()]
    return pd.DataFrame(rows).sort_values("Gap", ascending=False).reset_index(drop=True)


def recommended_skills(salary_data: pd.DataFrame, role: str, limit: int = 5) -> list[str]:
    skills = salary_data.loc[salary_data.job_role.eq(role), "skills"].dropna().str.split(",").explode().str.strip()
    return skills.value_counts().head(limit).index.tolist()


def india_entry_salary_ranges(salary_data: pd.DataFrame, roles: list[str]) -> pd.DataFrame:
    subset = salary_data.loc[(salary_data.country.eq("India")) & (salary_data.experience_level.eq("Entry")) & salary_data.job_role.isin(roles), ["job_role", "salary_usd"]]
    if subset.empty:
        return pd.DataFrame(columns=["Role", "Lower USD", "Upper USD", "Sample size"])
    result = subset.groupby("job_role").salary_usd.agg(**{"Lower USD": lambda x: x.quantile(.25), "Upper USD": lambda x: x.quantile(.75), "Sample size": "size"}).reset_index().rename(columns={"job_role": "Role"})
    result[["Lower USD", "Upper USD"]] = result[["Lower USD", "Upper USD"]].round().astype(int)
    return result.sort_values("Role").reset_index(drop=True)


def learning_plan(gaps: pd.DataFrame) -> list[tuple[str, list[str]]]:
    focus = ", ".join(gaps.loc[gaps.Gap.gt(0), "Area"].head(3)) or "interview practice and portfolio polish"
    return [("Week 1", [f"Strengthen {focus}", "Set a realistic study schedule"]), ("Week 2", ["Solve aptitude and problem-solving exercises", "Document learning on GitHub"]), ("Week 3", ["Build or improve one portfolio project", "Request feedback from a mentor or peer"]), ("Week 4", ["Tailor resume to the target role", "Practice mock interviews and applications"])]


def company_type_suitability(profile: dict) -> pd.DataFrame:
    """Explainable company-environment fit scores, not employer recommendations."""
    readiness = readiness_score(profile)
    scores = [
        ("Product companies", min(100, readiness + profile["coding_skills"] * 1.8 + profile["projects"] * 1.2), "Portfolio depth and technical strength"),
        ("Service companies", min(100, readiness + profile["communication_skills"] * 2.0 + profile["aptitude_score"] * .08), "Communication and problem solving"),
        ("Startups", min(100, readiness + profile["projects"] * 2.4 + profile["internships"] * 2.5), "Hands-on projects and practical exposure"),
    ]
    return pd.DataFrame(scores, columns=["Company type", "Suitability", "Why this fits"]).round({"Suitability": 1}).sort_values("Suitability", ascending=False).reset_index(drop=True)


def load_salary_data(project_root: Path) -> pd.DataFrame:
    return pd.read_csv(project_root / "data" / "raw" / "ai_job_salary_dataset_10k.csv")
