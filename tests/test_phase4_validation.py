import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "streamlit"))
sys.path.insert(0, str(ROOT))

from auth_store import authenticate_user, create_user
from career_intelligence import india_entry_salary_ranges, load_salary_data, readiness_score, role_matches, skill_gaps
from feature_engineering import MODEL_FEATURES, add_placement_features


FEATURES = ["gender", "age", "degree", "branch", "cgpa", "backlogs", "internships", "certifications", "coding_skills", "communication_skills", "aptitude_score", "projects"]
STRONG = {"gender": "Male", "age": 21, "degree": "BTech", "branch": "CS", "cgpa": 8.4, "backlogs": 0, "internships": 2, "certifications": 3, "coding_skills": 8, "communication_skills": 8, "aptitude_score": 82, "projects": 3}
DEVELOPING = STRONG | {"cgpa": 7.2, "internships": 1, "certifications": 1, "coding_skills": 6, "communication_skills": 6, "aptitude_score": 68, "projects": 1}
NEEDS_SUPPORT = STRONG | {"cgpa": 6.2, "backlogs": 2, "internships": 0, "certifications": 0, "coding_skills": 4, "communication_skills": 4, "aptitude_score": 55, "projects": 0}


class PhaseFourValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(ROOT / "models" / "placement_prediction_model.pkl")
        cls.salary_data = load_salary_data(ROOT)

    def test_profiles_fit_the_deployment_schema(self):
        rows = add_placement_features(pd.DataFrame([STRONG, DEVELOPING, NEEDS_SUPPORT]))
        probabilities = self.model.predict_proba(rows[MODEL_FEATURES])[:, 1]
        self.assertGreater(probabilities[0], 0.95)
        self.assertGreater(probabilities[1], probabilities[2])
        self.assertLess(probabilities[2], 0.05)

    def test_readiness_and_role_matches_are_bounded(self):
        for profile in (STRONG, DEVELOPING, NEEDS_SUPPORT):
            self.assertGreaterEqual(readiness_score(profile), 0)
            self.assertLessEqual(readiness_score(profile), 100)
            matches = role_matches(profile)
            self.assertEqual(len(matches), 8)
            self.assertTrue(matches["Profile match"].between(0, 100).all())
            self.assertFalse(skill_gaps(profile, matches.iloc[0]["Role"]).empty)

    def test_selected_technical_skills_change_the_top_role(self):
        analyst = DEVELOPING | {"technical_skills": ["SQL", "Excel", "Power BI", "Data Visualization"]}
        nlp = DEVELOPING | {"technical_skills": ["Python", "NLP", "Transformers", "PyTorch", "Deep Learning"]}
        self.assertEqual(role_matches(analyst).iloc[0]["Role"], "Data Analyst")
        self.assertEqual(role_matches(nlp).iloc[0]["Role"], "NLP Engineer")

    def test_salary_context_is_available_for_top_roles(self):
        top_roles = role_matches(DEVELOPING).head(5)["Role"].tolist()
        ranges = india_entry_salary_ranges(self.salary_data, top_roles)
        self.assertFalse(ranges.empty)
        self.assertTrue((ranges["Upper USD"] >= ranges["Lower USD"]).all())
        self.assertTrue((ranges["Sample size"] > 0).all())

    def test_local_accounts_never_store_plaintext_passwords(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "users.db"
            self.assertTrue(create_user(database, "Student Test", "student@example.com", "secure-pass-123")[0])
            self.assertEqual(authenticate_user(database, "student@example.com", "secure-pass-123"), (True, "Student Test"))
            self.assertFalse(authenticate_user(database, "student@example.com", "incorrect")[0])
            import sqlite3
            connection = sqlite3.connect(database)
            try:
                saved_hash = connection.execute("SELECT password_hash FROM users").fetchone()[0]
            finally:
                connection.close()
            self.assertNotIn("secure-pass-123", saved_hash)


if __name__ == "__main__":
    unittest.main()
