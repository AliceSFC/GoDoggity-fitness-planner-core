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


# ── Fitness Rule Definitions ────────────────────────────────────────

RULES = {
    "RULE-001": "Balanced training: mix of strength, flexibility, body awareness",
    "RULE-004": "Recovery: rest after sport day; rest after 2 consecutive training days",
    "RULE-005": "Minimum 1 rest day per week",
    "RULE-007": "Day after sport: only light exercises",
    "RULE-008": "Sport day: no fitness; day after: max light/recovery",
    "RULE-009": "Full body coverage: front + core + rear per week",
    "RULE-010": "All movement planes: sagittal + frontal + transverse per week",
    "RULE-012": "Warmup mandatory; stretching only after warmup",
}


# ── Classification Constants ────────────────────────────────────────

FOCUS_CATEGORIES = {
    "strength": {"strength", "power", "dynamic_strength"},
    "flexibility": {"flexibility", "rom", "lateral_flexibility", "hip_flexors", "stretching"},
    "body_awareness": {
        "body awareness", "body_awareness", "coordination",
        "mental control", "proprioception",
    },
}

BODY_REGION_KEYWORDS = {
    "front": {"front_end", "shoulders"},
    "core": {"core", "stabilization", "low_back"},
    "rear": {"rear_end", "rear_end_awareness", "hip_flexors"},
}

MOVEMENT_PLANE_KEYWORDS = {
    "sagittal": {"front_end", "rear_end", "strength", "eccentric", "plyometric", "gait_training"},
    "frontal": {"lateral", "lateral_work", "lateral_bend", "lateral_muscles", "abduction_adduction"},
    "transverse": {"rotation", "spine"},
}

BODY_FOCUS_ROTATION = [
    {"rear", "core"},               # day 1: rear end + core
    {"front", "flexibility"},       # day 2: front end + flexibility
    {"full_body", "body_awareness"},  # day 3: full body + body awareness
]

_BODY_FOCUS_MAP = {
    "rear": {"rear_end", "rear_end_awareness", "hip_flexors"},
    "front": {"front_end", "shoulders"},
    "core": {"core", "stabilization", "low_back"},
    "flexibility": {"flexibility", "rom", "lateral_flexibility", "hip_flexors", "stretching"},
    "full_body": {"full_body"},
    "body_awareness": {
        "body_awareness", "coordination", "mental control", "proprioception",
    },
}


# ── Helpers ─────────────────────────────────────────────────────────

def _tokenize(raw: str) -> set[str]:
    """Split comma-separated string into normalized token set."""
    if not isinstance(raw, str):
        return set()
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


def _is_stretch(row: pd.Series) -> bool:
    """Check if exercise is a stretching/flexibility exercise."""
    tags = str(row.get("tags", "")).lower()
    name = str(row.get("name_en", "")).lower()
    return "stretching" in tags or "stretch" in name


def _matches_body_focus(focus_str: str, body_focus: set[str]) -> bool:
    """Check if an exercise's focus matches the target body focus."""
    tokens = _tokenize(focus_str)
    for target in body_focus:
        keywords = _BODY_FOCUS_MAP.get(target, set())
        if tokens & keywords:
            return True
    return False


# ── Classification Functions ────────────────────────────────────────

def classify_focus(focus_str: str) -> set[str]:
    """Classify exercise into strength / flexibility / body_awareness."""
    tokens = _tokenize(focus_str)
    cats = set()
    for cat, keywords in FOCUS_CATEGORIES.items():
        if tokens & keywords:
            cats.add(cat)
    return cats or {"body_awareness"}


def classify_body_region(focus_str: str) -> set[str]:
    """Classify exercise into front / core / rear body regions."""
    tokens = _tokenize(focus_str)
    regions = set()
    for region, keywords in BODY_REGION_KEYWORDS.items():
        if tokens & keywords:
            regions.add(region)
    if "full_body" in tokens:
        regions = {"front", "core", "rear"}
    return regions or {"core"}


def classify_movement_plane(focus_str: str, tags_str: str = "") -> set[str]:
    """Classify exercise into sagittal / frontal / transverse planes."""
    tokens = _tokenize(focus_str) | _tokenize(tags_str)
    planes = set()
    for plane, keywords in MOVEMENT_PLANE_KEYWORDS.items():
        if tokens & keywords:
            planes.add(plane)
    if "full_body" in tokens:
        planes = {"sagittal", "frontal", "transverse"}
    return planes or {"sagittal"}


# ── Exercise Selection ──────────────────────────────────────────────

_OUTPUT_COLS = ["exercise_id", "name_en", "focus", "difficulty", "video_url"]


def _pick_stretch(exercises: pd.DataFrame, day_index: int) -> pd.DataFrame:
    """Pick one stretching exercise, varying by day_index."""
    stretch_mask = exercises.apply(_is_stretch, axis=1)
    stretches = exercises[stretch_mask]
    if stretches.empty:
        return exercises.head(0)
    offset = day_index % len(stretches)
    return stretches.iloc[offset:offset + 1]


def select_exercises_for_day(
    exercises: pd.DataFrame,
    day_index: int,
    body_focus: set[str],
    count: int = 3,
) -> list[dict]:
    """Select exercises for a training day with body-focus and offset variation.

    Applies: RULE-001 (balanced focus), RULE-009 (body regions), RULE-010 (planes).
    Returns ``count`` main exercises + 1 stretch at the end (RULE-012).
    """
    cols = [c for c in _OUTPUT_COLS if c in exercises.columns]
    if exercises.empty:
        return []

    df = exercises.copy()
    stretch_mask = df.apply(_is_stretch, axis=1)
    non_stretches = df[~stretch_mask]

    if non_stretches.empty:
        non_stretches = df  # fallback: use everything

    # Score by body-focus match (matching first, then rest)
    match_col = non_stretches["focus"].apply(
        lambda f: _matches_body_focus(f, body_focus)
    )
    matching = non_stretches[match_col]
    non_matching = non_stretches[~match_col]
    ordered = pd.concat([matching, non_matching])

    # Apply day-index offset for variety
    n = len(ordered)
    if n > 0:
        offset = day_index % n
        idx = list(ordered.index)
        ordered = ordered.loc[idx[offset:] + idx[:offset]]

    picked = ordered.head(count)

    # Add stretch at end (RULE-012: stretching after warmup, never first)
    stretch = _pick_stretch(df, day_index)
    if not stretch.empty:
        # Avoid duplicate
        picked_ids = set(picked["exercise_id"]) if "exercise_id" in picked.columns else set()
        stretch_id = stretch.iloc[0].get("exercise_id", "")
        if stretch_id and stretch_id in picked_ids:
            # Already included — move it to end
            picked = picked[picked["exercise_id"] != stretch_id]
            picked = pd.concat([picked.head(count), stretch])
        else:
            picked = pd.concat([picked, stretch])

    return picked[cols].fillna("").to_dict(orient="records")


def select_light_exercises(
    exercises: pd.DataFrame,
    day_index: int,
    count: int = 4,
) -> list[dict]:
    """Select light exercises for recovery / post-sport days.

    Applies: RULE-007 (light only), RULE-008 (no intensive after sport).
    Only beginner/intermediate, prefers low impact & flexibility/body-awareness.
    Returns ``count`` main exercises + 1 stretch at end.
    """
    cols = [c for c in _OUTPUT_COLS if c in exercises.columns]
    if exercises.empty:
        return []

    df = exercises.copy()

    # Filter: only beginner / intermediate (no advanced / hard)
    if "difficulty" in df.columns:
        easy = df[df["difficulty"].astype(str).str.lower().isin(
            ["beginner", "intermediate"]
        )]
        if not easy.empty:
            df = easy

    # Prefer low impact
    if "impact" in df.columns:
        low = df[df["impact"].astype(str).str.lower() == "low"]
        if len(low) >= count:
            df = low

    # Separate stretches
    stretch_mask = df.apply(_is_stretch, axis=1)
    non_stretches = df[~stretch_mask]

    if non_stretches.empty:
        non_stretches = df

    # Prefer flexibility / body-awareness focus
    light_focus = {"flexibility", "body_awareness", "mental control", "coordination"}
    priority_col = non_stretches["focus"].apply(
        lambda f: bool(_tokenize(f) & light_focus)
    )
    prioritized = pd.concat([
        non_stretches[priority_col],
        non_stretches[~priority_col],
    ])

    # Apply offset for variety
    n = len(prioritized)
    if n > 0:
        offset = day_index % n
        idx = list(prioritized.index)
        prioritized = prioritized.loc[idx[offset:] + idx[:offset]]

    picked = prioritized.head(count)

    # Add stretch at end
    stretch = _pick_stretch(df, day_index)
    if not stretch.empty:
        picked_ids = set(picked["exercise_id"]) if "exercise_id" in picked.columns else set()
        stretch_id = stretch.iloc[0].get("exercise_id", "")
        if stretch_id and stretch_id in picked_ids:
            picked = picked[picked["exercise_id"] != stretch_id]
            picked = pd.concat([picked.head(count), stretch])
        else:
            picked = pd.concat([picked, stretch])

    return picked[cols].fillna("").to_dict(orient="records")
