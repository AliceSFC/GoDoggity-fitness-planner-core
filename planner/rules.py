import pandas as pd

DAY_MAP = {
    "mon": "Mon", "tue": "Tue", "wed": "Wed",
    "thu": "Thu", "fri": "Fri", "sat": "Sat", "sun": "Sun"
}

def parse_limitations(raw: str) -> set[str]:
    if not isinstance(raw, str) or not raw.strip():
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}

def parse_activities(raw: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if not isinstance(raw, str) or not raw.strip():
        return mapping

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        day_raw, act_raw = part.split(":", 1)
        day_norm = DAY_MAP.get(day_raw.strip().lower()[:3])
        if not day_norm:
            continue
        mapping.setdefault(day_norm, []).append(act_raw.strip().lower())
    return mapping

def filter_exercises(
    exercises: pd.DataFrame,
    limitations: set[str],
    age_group: str,
    equipment_available: str,
) -> pd.DataFrame:
    df = exercises.copy()

    # Senior safety
    if str(age_group).strip().lower() == "senior" and "senior_safe" in df.columns:
        df = df[df["senior_safe"].astype(str).str.upper() == "TRUE"]

    # Limitations
    if "no_balance" in limitations and "equipment" in df.columns:
        df = df[df["equipment"].astype(str).str.lower() != "balance"]

    if "low_impact" in limitations and "impact" in df.columns:
        df = df[df["impact"].astype(str).str.lower() == "low"]

    # Equipment available
    if isinstance(equipment_available, str) and equipment_available.strip().lower() == "none" and "equipment" in df.columns:
        df = df[df["equipment"].astype(str).str.lower() == "none"]

    return df
