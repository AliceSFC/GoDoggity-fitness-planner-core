import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def parse_limitations(raw: str) -> set[str]:
    if not isinstance(raw, str) or not raw.strip():
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def parse_activities(raw: str) -> dict[str, list[str]]:
    """
    Expected format: "tue:agility,thu:walk" (comma-separated, day:activity)
    Returns: {"Tue": ["agility"], "Thu": ["walk"]}
    """
    mapping = {}
    if not isinstance(raw, str) or not raw.strip():
        return mapping

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        day_raw, act_raw = part.split(":", 1)
        day = day_raw.strip().lower()[:3]
        act = act_raw.strip().lower()

        day_map = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat", "sun": "Sun"}
        day_norm = day_map.get(day)
        if not day_norm:
            continue

        mapping.setdefault(day_norm, []).append(act)
    return mapping



def load_data():
    exercises = pd.read_csv(DATA_DIR / "exercises.csv")
    cases = pd.read_csv(DATA_DIR / "test_cases.csv")
    return exercises, cases


def filter_exercises(exercises: pd.DataFrame, limitations: set[str], age_group: str, equipment_available: str):
    df = exercises.copy()

    # Age safety rule (very basic v1)
    if age_group.strip().lower() == "senior":
        df = df[df["senior_safe"].astype(str).str.upper() == "TRUE"]

    # Limitation rules (v1)
    if "no_balance" in limitations:
        df = df[df["equipment"].astype(str).str.lower() != "balance"]
    if "low_impact" in limitations:
        df = df[df["impact"].astype(str).str.lower() == "low"]

    # Equipment availability (v1)
    if isinstance(equipment_available, str) and equipment_available.strip().lower() == "none":
        df = df[df["equipment"].astype(str).str.lower() == "none"]

    return df


def make_week_plan(case_row, allowed_exercises: pd.DataFrame):
    week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    activities = parse_activities(case_row.get("activities", ""))
    sport_days = {d for d, acts in activities.items() if any(a in ["agility", "competition", "trial", "training"] for a in acts)}

    fitness = str(case_row.get("fitness_level", "medium")).lower()
    freq = {"low": 2, "medium": 3, "high": 4}.get(fitness, 3)

    candidate_days = [d for d in week if d not in sport_days]
    training_days = candidate_days[:freq]


    picked_rows = allowed_exercises.head(3)[
        ["exercise_id", "name_en", "focus", "difficulty", "video_url"]
    ].fillna("")

    exercises_out = picked_rows.to_dict(orient="records")

    plan = []
    for d in week:
        if d in training_days:
            plan.append(
                {
                    "day": d,
                    "type": "training",
                    "focus": "mixed",
                    "warmup": "5 min easy walking + gentle mobility",
                    "exercises": exercises_out,
                    "cooldown": "2–5 min calm walking",
                }
            )
        else:
            plan.append({"day": d, "type": "rest", "note": "Rest day (walking is OK)"})

# Ensure at least one full rest day (no sport activity + no fitness)
full_rest_candidates = [d for d in week if d not in training_days and d not in sport_days]
if not full_rest_candidates:
    # If everything is filled (rare), drop the last training day
    if training_days:
        training_days = training_days[:-1]


    return plan



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", required=True)
    args = parser.parse_args()

    exercises, cases = load_data()
    case = cases[cases["case_id"] == args.case_id]
    if case.empty:
        raise SystemExit(f"Unknown case_id: {args.case_id}")

    row = case.iloc[0].to_dict()

    limitations = parse_limitations(row.get("limitations", ""))
    age_group = str(row.get("age_group", "adult"))
    equipment_available = str(row.get("equipment_available", "none"))

    allowed = filter_exercises(exercises, limitations, age_group, equipment_available)
    week_plan = make_week_plan(row, allowed)

    output = {"case_id": args.case_id, "dog_name": row.get("dog_name"), "plan": week_plan}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
