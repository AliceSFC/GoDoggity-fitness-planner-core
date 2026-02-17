import pandas as pd
from .rules import parse_activities

WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SPORT_TAGS = {"agility", "competition", "trial", "training"}


def _get_training_range(fitness_level: str) -> tuple[int, int]:
    """Return (min, max) training days per week based on fitness level."""
    ranges = {"low": (1, 2), "medium": (2, 3), "high": (3, 4)}
    return ranges.get(fitness_level, (2, 3))


def _find_sport_days(activities: dict[str, list[str]]) -> set[str]:
    """Identify days with heavy sport activities."""
    return {
        d for d, acts in activities.items()
        if any(a in SPORT_TAGS for a in acts)
    }


def _place_training_days(
    open_days: list[str], target: int, has_sport: bool
) -> list[str]:
    """Place training days with even spacing, respecting constraints.

    Strategy:
    1. Try ideal spacing (based on open slots / target)
    2. Fall back to minimum gap of 2 days
    3. Last resort (no sport only): allow consecutive
    """
    if not open_days or target <= 0:
        return []
    target = min(target, len(open_days))
    if target == len(open_days) and not has_sport:
        return list(open_days)

    ideal_gap = len(open_days) / target
    min_gap = max(2, int(ideal_gap))

    # First pass: place with ideal spacing
    placed = []
    for d in open_days:
        if len(placed) >= target:
            break
        if placed:
            gap = WEEK.index(d) - WEEK.index(placed[-1])
            if gap < min_gap:
                continue
        placed.append(d)

    # Second pass: relax to min_gap=2 if not enough
    if len(placed) < target:
        placed = []
        for d in open_days:
            if len(placed) >= target:
                break
            if placed:
                gap = WEEK.index(d) - WEEK.index(placed[-1])
                if gap < 2:
                    continue
            placed.append(d)

    # Third pass: allow consecutive only when no sport (last resort)
    if len(placed) < target and not has_sport:
        for d in open_days:
            if len(placed) >= target:
                break
            if d not in placed:
                placed.append(d)
        placed.sort(key=lambda x: WEEK.index(x))

    return placed


def _build_schedule(
    sport_days: set[str], training_range: tuple[int, int], has_sport: bool
) -> dict[str, str]:
    """Build weekly schedule respecting all placement rules.

    Rules enforced:
    - Rest day after each sport day (heavy -> recovery)
    - At least 1 rest day per week
    - No 2 consecutive rest days (use 'light' instead)
    - No 2 consecutive training days when sport is in the week
    - Training count within fitness-level range
    """
    schedule = {d: None for d in WEEK}

    # Step 1: fix sport days
    for d in sport_days:
        schedule[d] = "sport"

    # Step 2: mandatory rest after sport days
    # (if next day is already sport, we accept it — user's fixed schedule)
    for i, d in enumerate(WEEK):
        if schedule[d] == "sport" and i + 1 < len(WEEK):
            next_d = WEEK[i + 1]
            if schedule[next_d] is None:
                schedule[next_d] = "rest"

    # Step 3: place training days in remaining open slots
    open_days = [d for d in WEEK if schedule[d] is None]
    _, max_train = training_range
    training_days = _place_training_days(open_days, max_train, has_sport)

    for d in training_days:
        schedule[d] = "training"

    # Step 4: fill remaining as rest
    for d in WEEK:
        if schedule[d] is None:
            schedule[d] = "rest"

    # Step 5: no 2 consecutive rest days -> convert second to light
    for i in range(len(WEEK) - 1):
        if schedule[WEEK[i]] == "rest" and schedule[WEEK[i + 1]] == "rest":
            schedule[WEEK[i + 1]] = "light"

    # Step 6: ensure at least 1 full rest day
    rest_count = sum(1 for v in schedule.values() if v == "rest")
    if rest_count < 1:
        training_in_sched = [d for d in WEEK if schedule[d] == "training"]
        for d in reversed(training_in_sched):
            idx = WEEK.index(d)
            prev_rest = idx > 0 and schedule[WEEK[idx - 1]] == "rest"
            next_rest = idx < 6 and schedule[WEEK[idx + 1]] == "rest"
            if not prev_rest and not next_rest:
                schedule[d] = "rest"
                break
        else:
            if training_in_sched:
                schedule[training_in_sched[-1]] = "rest"

    return schedule


def make_week_plan(
    case_row: dict, allowed_exercises: pd.DataFrame
) -> list:
    fitness = str(case_row.get("fitness_level", "medium")).lower()
    training_range = _get_training_range(fitness)

    activities = parse_activities(case_row.get("activities", ""))
    sport_days = _find_sport_days(activities)
    has_sport = len(sport_days) > 0

    schedule = _build_schedule(sport_days, training_range, has_sport)

    # Pick exercises (same selection as before)
    cols = ["exercise_id", "name_en", "focus", "difficulty", "video_url"]
    existing_cols = [c for c in cols if c in allowed_exercises.columns]
    picked = (
        allowed_exercises.head(3)[existing_cols]
        .fillna("")
        .to_dict(orient="records")
    )

    plan = []
    for d in WEEK:
        day_type = schedule[d]
        if day_type == "sport":
            plan.append({
                "day": d,
                "type": "sport_only",
                "note": (
                    f"Planned activity: {', '.join(activities.get(d, []))}."
                    " No extra fitness today."
                ),
            })
        elif day_type == "training":
            plan.append({
                "day": d,
                "type": "training",
                "focus": "mixed",
                "warmup": "5 min easy walking + gentle mobility",
                "exercises": picked,
                "cooldown": "2–5 min calm walking",
            })
        elif day_type == "light":
            plan.append({
                "day": d,
                "type": "light",
                "note": "Light activity day (easy walk, gentle stretching)",
            })
        else:
            plan.append({
                "day": d,
                "type": "rest",
                "note": "Rest day (walking is OK)",
            })

    return plan
