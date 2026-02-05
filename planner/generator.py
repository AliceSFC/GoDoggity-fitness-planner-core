import pandas as pd
from .rules import parse_activities

WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SPORT_TAGS = {"agility", "competition", "trial", "training"}

def make_week_plan(case_row: dict, allowed_exercises: pd.DataFrame):
    fitness = str(case_row.get("fitness_level", "medium")).lower()
    freq = {"low": 2, "medium": 3, "high": 4}.get(fitness, 3)

    activities = parse_activities(case_row.get("activities", ""))
    sport_days = {
        d for d, acts in activities.items()
        if any(a in SPORT_TAGS for a in acts)
    }

    # Basic scheduling: avoid sport days
    candidate_days = [d for d in WEEK if d not in sport_days]
    training_days = candidate_days[:freq]

    # Pick top 2–4 exercises (simple)
    cols = ["exercise_id", "name_en", "focus", "difficulty", "video_url"]
    existing_cols = [c for c in cols if c in allowed_exercises.columns]
    picked = allowed_exercises.head(3)[existing_cols].fillna("").to_dict(orient="records")

    plan = []
    for d in WEEK:
        if d in sport_days:
            plan.append({
                "day": d,
                "type": "sport_only",
                "note": f"Planned activity: {', '.join(activities.get(d, []))}. No extra fitness today."
            })
        elif d in training_days:
            plan.append({
                "day": d,
                "type": "training",
                "focus": "mixed",
                "warmup": "5 min easy walking + gentle mobility",
                "exercises": picked,
                "cooldown": "2–5 min calm walking",
            })
        else:
            plan.append({
                "day": d,
                "type": "rest",
                "note": "Rest day (walking is OK)"
            })
    return plan
