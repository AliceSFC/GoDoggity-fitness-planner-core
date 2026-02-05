from fastapi import FastAPI, HTTPException

from planner.io import load_data
from planner.rules import parse_limitations, filter_exercises
from planner.generator import make_week_plan

app = FastAPI(title="GoDoggity Fitness Planner Core")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/generate-plan")
def generate_plan(case_id: str):
    exercises, cases = load_data()

    case = cases[cases["case_id"] == case_id]
    if case.empty:
        raise HTTPException(status_code=404, detail="Unknown case_id")

    row = case.iloc[0].to_dict()

    limitations = parse_limitations(row.get("limitations", ""))
    age_group = str(row.get("age_group", "adult"))
    equipment_available = str(row.get("equipment_available", "none"))

    allowed = filter_exercises(exercises, limitations, age_group, equipment_available)
    plan = make_week_plan(row, allowed)

    return {"case_id": case_id, "dog_name": row.get("dog_name"), "plan": plan}
