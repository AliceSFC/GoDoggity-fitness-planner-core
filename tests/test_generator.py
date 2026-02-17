import pandas as pd
import pytest
from planner.generator import (
    make_week_plan,
    _build_schedule,
    _find_sport_days,
    _get_training_range,
    WEEK,
)


def _dummy_exercises():
    """Minimal exercise DataFrame for testing."""
    return pd.DataFrame([
        {"exercise_id": "EX_001", "name_en": "Exercise A", "focus": "core",
         "difficulty": "beginner", "video_url": ""},
        {"exercise_id": "EX_002", "name_en": "Exercise B", "focus": "strength",
         "difficulty": "intermediate", "video_url": ""},
        {"exercise_id": "EX_003", "name_en": "Exercise C", "focus": "coordination",
         "difficulty": "beginner", "video_url": ""},
    ])


def _get_types(plan):
    """Extract {day: type} mapping from plan output."""
    return {entry["day"]: entry["type"] for entry in plan}


# ── Helper function tests ────────────────────────────────────────────

class TestTrainingRange:
    def test_low(self):
        assert _get_training_range("low") == (1, 2)

    def test_medium(self):
        assert _get_training_range("medium") == (2, 3)

    def test_high(self):
        assert _get_training_range("high") == (3, 4)

    def test_unknown_defaults_to_medium(self):
        assert _get_training_range("unknown") == (2, 3)


class TestFindSportDays:
    def test_agility_is_sport(self):
        assert _find_sport_days({"Mon": ["agility"]}) == {"Mon"}

    def test_competition_is_sport(self):
        assert _find_sport_days({"Sat": ["competition"]}) == {"Sat"}

    def test_walk_is_not_sport(self):
        assert _find_sport_days({"Wed": ["walk"]}) == set()

    def test_mixed_activities(self):
        acts = {"Mon": ["agility"], "Wed": ["walk"], "Fri": ["trial"]}
        assert _find_sport_days(acts) == {"Mon", "Fri"}

    def test_empty(self):
        assert _find_sport_days({}) == set()


# ── Schedule rule validation ─────────────────────────────────────────

class TestScheduleRules:
    """Verify _build_schedule enforces every rule across many combos."""

    SPORT_COMBOS = [
        set(),
        {"Tue"},
        {"Mon", "Thu"},
        {"Mon", "Wed", "Fri"},
        {"Mon", "Tue"},            # consecutive sport
        {"Sat"},                   # sport near end of week
    ]
    FITNESS_LEVELS = ["low", "medium", "high"]

    def _all_schedules(self):
        """Yield (schedule, sport_set, fitness) for every combo."""
        for sport in self.SPORT_COMBOS:
            for fit in self.FITNESS_LEVELS:
                tr = _get_training_range(fit)
                sched = _build_schedule(sport, tr, len(sport) > 0)
                yield sched, sport, fit

    # Rule 3: rest or light after sport
    def test_rest_after_sport_day(self):
        for sched, sport, fit in self._all_schedules():
            for i, d in enumerate(WEEK):
                if sched[d] == "sport" and i + 1 < len(WEEK):
                    next_type = sched[WEEK[i + 1]]
                    assert next_type in ("rest", "light", "sport"), (
                        f"After sport on {d}, got '{next_type}' "
                        f"(sport={sport}, fitness={fit})"
                    )

    # Rule 4: at least 1 rest day
    def test_at_least_one_rest_day(self):
        for sched, sport, fit in self._all_schedules():
            rest = sum(1 for v in sched.values() if v == "rest")
            assert rest >= 1, (
                f"No rest day: sport={sport}, fitness={fit}, "
                f"schedule={list(sched.values())}"
            )

    # Rule 5: no 2 consecutive rest days
    def test_no_consecutive_rest_days(self):
        for sched, sport, fit in self._all_schedules():
            for i in range(len(WEEK) - 1):
                pair = (sched[WEEK[i]], sched[WEEK[i + 1]])
                assert pair != ("rest", "rest"), (
                    f"Consecutive rest {WEEK[i]}-{WEEK[i+1]}: "
                    f"sport={sport}, fitness={fit}"
                )

    # Rule 6: no 2 consecutive training when sport in week
    def test_no_consecutive_training_with_sport(self):
        for sched, sport, fit in self._all_schedules():
            if not sport:
                continue
            for i in range(len(WEEK) - 1):
                both_train = (
                    sched[WEEK[i]] == "training"
                    and sched[WEEK[i + 1]] == "training"
                )
                assert not both_train, (
                    f"Consecutive training {WEEK[i]}-{WEEK[i+1]} "
                    f"with sport={sport}, fitness={fit}"
                )

    # Training-count ranges (no-sport only, sport reduces availability)
    def test_training_count_no_sport(self):
        for fit in self.FITNESS_LEVELS:
            lo, hi = _get_training_range(fit)
            sched = _build_schedule(set(), (lo, hi), has_sport=False)
            count = sum(1 for v in sched.values() if v == "training")
            assert lo <= count <= hi, (
                f"fitness={fit}: expected {lo}-{hi} training days, got {count}"
            )

    # Every day accounted for
    def test_all_days_assigned(self):
        for sched, sport, fit in self._all_schedules():
            for d in WEEK:
                assert sched[d] in ("sport", "training", "rest", "light"), (
                    f"{d} unassigned: sport={sport}, fitness={fit}"
                )


# ── Integration tests (make_week_plan) ───────────────────────────────

class TestMakeWeekPlan:
    def test_high_fitness_one_sport_day(self):
        """Chouffe scenario: high fitness, agility on Tuesday."""
        case = {"fitness_level": "high", "activities": "tue:agility,thu:walk"}
        plan = make_week_plan(case, _dummy_exercises())
        types = _get_types(plan)

        assert len(plan) == 7
        assert types["Tue"] == "sport_only"
        assert types["Wed"] in ("rest", "light")
        training_count = sum(1 for v in types.values() if v == "training")
        assert training_count >= 3

    def test_medium_fitness_no_sport(self):
        case = {"fitness_level": "medium", "activities": ""}
        plan = make_week_plan(case, _dummy_exercises())
        types = _get_types(plan)

        training = sum(1 for v in types.values() if v == "training")
        rest = sum(1 for v in types.values() if v == "rest")
        assert 2 <= training <= 3
        assert rest >= 1

    def test_low_fitness_many_sport_days(self):
        case = {
            "fitness_level": "low",
            "activities": "mon:agility,wed:competition,fri:training",
        }
        plan = make_week_plan(case, _dummy_exercises())
        types = _get_types(plan)

        sport = sum(1 for v in types.values() if v == "sport_only")
        assert sport == 3
        training = sum(1 for v in types.values() if v == "training")
        assert training >= 1

    def test_consecutive_sport_days(self):
        case = {
            "fitness_level": "medium",
            "activities": "mon:agility,tue:competition",
        }
        plan = make_week_plan(case, _dummy_exercises())
        types = _get_types(plan)

        assert types["Mon"] == "sport_only"
        assert types["Tue"] == "sport_only"
        assert types["Wed"] in ("rest", "light")

    def test_no_activities_at_all(self):
        case = {"fitness_level": "medium"}
        plan = make_week_plan(case, _dummy_exercises())
        types = _get_types(plan)

        assert len(plan) == 7
        training = sum(1 for v in types.values() if v == "training")
        assert 2 <= training <= 3

    def test_sport_on_sunday(self):
        """Sport on last day — no 'next day' to place rest."""
        case = {"fitness_level": "medium", "activities": "sun:agility"}
        plan = make_week_plan(case, _dummy_exercises())
        types = _get_types(plan)

        assert types["Sun"] == "sport_only"

    def test_output_structure_training(self):
        case = {"fitness_level": "high", "activities": ""}
        plan = make_week_plan(case, _dummy_exercises())
        t_days = [e for e in plan if e["type"] == "training"]

        assert len(t_days) >= 1
        day = t_days[0]
        assert "warmup" in day
        assert "exercises" in day
        assert isinstance(day["exercises"], list)
        assert len(day["exercises"]) == 3
        assert "cooldown" in day

    def test_output_structure_sport(self):
        case = {"fitness_level": "high", "activities": "tue:agility"}
        plan = make_week_plan(case, _dummy_exercises())
        s_days = [e for e in plan if e["type"] == "sport_only"]

        assert len(s_days) == 1
        assert "agility" in s_days[0]["note"]

    def test_output_structure_light(self):
        """Light days should have a note."""
        case = {"fitness_level": "low", "activities": ""}
        plan = make_week_plan(case, _dummy_exercises())
        l_days = [e for e in plan if e["type"] == "light"]

        # Low fitness with 2 training -> multiple rest -> some become light
        for day in l_days:
            assert "note" in day

    def test_all_seven_days_present(self):
        case = {"fitness_level": "medium", "activities": ""}
        plan = make_week_plan(case, _dummy_exercises())
        days = [e["day"] for e in plan]
        assert days == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def test_empty_exercises_dataframe(self):
        """Should not crash with no available exercises."""
        case = {"fitness_level": "medium", "activities": ""}
        empty_df = pd.DataFrame(columns=["exercise_id", "name_en", "focus",
                                          "difficulty", "video_url"])
        plan = make_week_plan(case, empty_df)
        t_days = [e for e in plan if e["type"] == "training"]
        assert all(d["exercises"] == [] for d in t_days)
