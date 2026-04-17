import kaggle_benchmarks as kbench
import pandas as pd
import time

scenarios = [
    {"id":1, "task":"Pick the red block and place it in zone A.", "completed":["Moved to red block","Picked up red block"], "failure":"Red block slipped from gripper.", "constraint":"Carry one object at a time.", "bad_steps":["Moved to red block","Picked up red block"], "max_steps":4, "difficulty":"easy"},
    {"id":2, "task":"Pick the blue block and place it in zone B.", "completed":["Moved to blue block","Picked up blue block"], "failure":"Zone B is blocked by an obstacle.", "constraint":"Carry one object at a time.", "bad_steps":["Moved to blue block","Picked up blue block"], "max_steps":4, "difficulty":"easy"},
    {"id":3, "task":"Pick the green block and place it in zone C.", "completed":["Moved to green block"], "failure":"Green block not found at expected location.", "constraint":"Carry one object at a time.", "bad_steps":["Moved to green block"], "max_steps":4, "difficulty":"easy"},
    {"id":4, "task":"Stack red block on blue block, then move stack to zone A.", "completed":["Picked up red block","Placed red block on blue block"], "failure":"Stack fell over.", "constraint":"Carry one object at a time. You CANNOT lift a stack — move each block separately.", "bad_steps":["Picked up red block","Placed red block on blue block"], "max_steps":5, "difficulty":"medium"},
    {"id":5, "task":"Move red block to zone B, then green block to zone B.", "completed":["Picked up red block","Placed red block in zone B"], "failure":"Green block is under heavy object.", "constraint":"Carry one object at a time. You cannot move the heavy object.", "bad_steps":["Picked up red block","Placed red block in zone B"], "max_steps":4, "difficulty":"medium"},
    {"id":6, "task":"Pick blue block and place on shelf.", "completed":["Moved to blue block","Picked up blue block","Moved to shelf"], "failure":"Shelf is at max capacity.", "constraint":"Carry one object at a time. Shelf cannot accept more items.", "bad_steps":["Moved to blue block","Picked up blue block","Moved to shelf"], "max_steps":4, "difficulty":"medium"},
    {"id":7, "task":"Pick red block, place in zone A, then pick blue block, place in zone A.", "completed":["Picked red block","Placed red block in zone A","Moved to blue block","Picked up blue block"], "failure":"Zone A is now full — cannot place anything else there.", "constraint":"Carry one object at a time. Zone A cannot accept more items.", "bad_steps":["Picked red block","Placed red block in zone A","Moved to blue block","Picked up blue block"], "max_steps":3, "difficulty":"hard"},
    {"id":8, "task":"Pick red block and place in zone C.", "completed":["Moved to red block","Picked up red block","Moved to zone C"], "failure":"Zone C floor sensor triggered — unsafe to place. Zone C is permanently closed.", "constraint":"Carry one object at a time. Do NOT place anything in zone C.", "bad_steps":["Moved to red block","Picked up red block","Moved to zone C"], "max_steps":4, "difficulty":"hard"},
    {"id":9, "task":"Stack green on blue, then stack red on green.", "completed":["Picked green block","Placed green on blue"], "failure":"Blue block tipped over — entire stack must be rebuilt from scratch.", "constraint":"Carry one object at a time. Rebuild order must be: blue first, then green, then red.", "bad_steps":["Picked green block","Placed green on blue"], "max_steps":6, "difficulty":"hard"},
    {"id":10, "task":"Pick blue block and place in zone D.", "completed":["Moved to blue block","Picked up blue block","Moved to zone D"], "failure":"Zone D does not exist — it was a map error. Use zone B instead.", "constraint":"Carry one object at a time. Zone D does not exist.", "bad_steps":["Moved to blue block","Picked up blue block","Moved to zone D"], "max_steps":3, "difficulty":"hard"},
    {"id":11, "task":"Pick yellow cube, place in zone B, return to home.", "completed":["Picked yellow cube","Placed in zone B"], "failure":"Return path blocked by fallen object. Home position also moved — new home is zone E.", "constraint":"Carry one object at a time. Do not pick up the fallen object. New home is zone E.", "bad_steps":["Picked yellow cube","Placed in zone B"], "max_steps":3, "difficulty":"very_hard"},
    {"id":12, "task":"Pick red block, place in zone A. Then pick green block, place in zone A.", "completed":["Picked red block","Placed red block in zone A"], "failure":"Red block placed incorrectly — must be repositioned to zone B before continuing.", "constraint":"Carry one object at a time. Red block must be in zone B before green block is moved.", "bad_steps":["Picked red block","Placed red block in zone A"], "max_steps":5, "difficulty":"very_hard"},
    {"id":13, "task":"Pick green block and place on shelf.", "completed":["Moved to green block","Picked up green block"], "failure":"Shelf collapsed — no shelf available. Must place on floor in zone C instead.", "constraint":"Carry one object at a time. Shelf no longer exists.", "bad_steps":["Moved to green block","Picked up green block"], "max_steps":3, "difficulty":"medium"},
    {"id":14, "task":"Pick red block and place in zone A.", "completed":[], "failure":"Red block location unknown — sensor failure. Last known location was zone B area.", "constraint":"Carry one object at a time. Search zone B area first.", "bad_steps":[], "max_steps":5, "difficulty":"medium"},
    {"id":15, "task":"Move blue block to zone C, then move green block to zone C.", "completed":["Picked blue block","Placed blue block in zone C","Moved to green block"], "failure":"Gripper dropped green block mid-transit — green block now in unknown location.", "constraint":"Carry one object at a time. Must locate green block before picking it up.", "bad_steps":["Picked blue block","Placed blue block in zone C","Moved to green block"], "max_steps":4, "difficulty":"hard"},
    {"id":16, "task":"Pick the red block and place it in zone A.", "completed":["Moved to red block","Picked up red block","Moved to zone A","Placed red block in zone A"], "failure":"Post-placement sensor shows red block is correctly placed. No issues detected.", "constraint":"Carry one object at a time.", "bad_steps":["Moved to red block","Picked up red block","Moved to zone A","Placed red block in zone A"], "max_steps":1, "difficulty":"trick"},
    {"id":17, "task":"Pick yellow cube and stack it on red block.", "completed":["Moved to yellow cube","Picked up yellow cube","Moved to red block"], "failure":"Red block moved by external force — now in zone D. Yellow cube still in gripper.", "constraint":"Carry one object at a time. Yellow cube is currently held.", "bad_steps":["Moved to yellow cube","Picked up yellow cube","Moved to red block"], "max_steps":3, "difficulty":"hard"},
    {"id":18, "task":"Pick green block, place in zone C, then return to home.", "completed":["Picked up green block","Placed green block in zone C"], "failure":"Home position blocked by another robot that will not move.", "constraint":"Carry one object at a time. Cannot move the other robot. Find an alternative wait position.", "bad_steps":["Picked up green block","Placed green block in zone C"], "max_steps":3, "difficulty":"very_hard"},
    {"id":19, "task":"Pick red block and place in zone B.", "completed":["Moved to red block","Picked up red block","Moved to zone B"], "failure":"Zone B floor sensor triggered — unsafe. Zone A also unsafe. Only zone C is safe.", "constraint":"Carry one object at a time. Only zone C is safe to place objects.", "bad_steps":["Moved to red block","Picked up red block","Moved to zone B"], "max_steps":3, "difficulty":"very_hard"},
    {"id":20, "task":"Pick blue block and place in zone A, then pick red block and place in zone A.", "completed":["Picked blue block","Placed blue in zone A","Moved to red block"], "failure":"Red block is glued to floor — cannot be picked up.", "constraint":"Carry one object at a time. Red block cannot be moved under any circumstances.", "bad_steps":["Picked blue block","Placed blue in zone A","Moved to red block"], "max_steps":1, "difficulty":"trick"},
]

df = pd.DataFrame(scenarios)

# ── SCORING CONSTANTS ─────────────────────────────────────────────────────────
DIFFICULTY_WEIGHTS = {
    "easy": 1.0,
    "medium": 2.0,
    "hard": 3.0,
    "very_hard": 4.0,
    "trick": 5.0
}
TOTAL_CHECKS = 9  # 4 hard checks + 5 judge criteria

def speed_bonus(seconds):
    if seconds < 3:   return 0.3
    if seconds < 10:  return 0.2
    if seconds < 30:  return 0.1
    return 0.0

def verbosity_bonus(word_count):
    if word_count < 80:  return 0.2
    if word_count < 150: return 0.1
    return 0.0

@kbench.task(
    name="robot-replanning-executive-functions",
    description="Tests executive functions in LLMs: weighted scoring by difficulty, speed bonus, verbosity bonus, and strict 90% accuracy threshold across 20 robot replanning scenarios."
)
def robot_replan(llm, id, task, completed, failure, constraint, bad_steps, max_steps, difficulty) -> float:

    completed_str = "\n".join(f"  - {s}" for s in completed) if completed else "  (none yet)"

    prompt = f"""You are controlling a robot arm. A task was in progress but a failure occurred.

ORIGINAL TASK: {task}
CONSTRAINT: {constraint}

STEPS ALREADY COMPLETED (do NOT repeat these):
{completed_str}

FAILURE: {failure}

Write a new plan to complete the remaining goal.
- Number each step (1. 2. 3. ...)
- Do NOT include any already-completed steps
- Strictly respect ALL constraints
- Be as concise and efficient as possible"""

    # ── MEASURE SPEED ─────────────────────────────────────────────────────────
    start = time.time()
    response = llm.prompt(prompt)
    elapsed = time.time() - start

    # ── MEASURE VERBOSITY ─────────────────────────────────────────────────────
    word_count = len(response.split())
    numbered_lines = [l.strip() for l in response.strip().split('\n')
                      if l.strip() and l.strip()[0].isdigit()]
    step_count = len(numbered_lines)

    # ── TRACK CHECKS PASSED ───────────────────────────────────────────────────
    checks_passed = 0

    # HARD CHECK 1: Numbered list
    has_numbers = bool(__import__('re').search(r'\d+\.', response))
    if has_numbers:
        checks_passed += 1

    # HARD CHECK 2: No repeated steps
    no_repeats = True
    for step in bad_steps:
        key_words = [w for w in step.lower().split() if len(w) > 4]
        for word in key_words:
            if response.lower().count(word) > 1:
                no_repeats = False
    if no_repeats:
        checks_passed += 1

    # HARD CHECK 3: Efficient step count
    if step_count <= max_steps + 1:
        checks_passed += 1

    # HARD CHECK 4: No constraint violation
    no_violation = (
        "carry both" not in response.lower() and
        "pick up both" not in response.lower() and
        "grab both" not in response.lower()
    )
    if no_violation:
        checks_passed += 1

    # JUDGE CHECKS 5-9: 5 criteria, each counts as 1
    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=kbench.judge_llm,
        criteria=[
            "The plan must NOT repeat any step that was listed as already completed.",
            "The plan must directly address the failure and propose a concrete alternative.",
            f"The plan must strictly respect this constraint: {constraint}",
            "The plan must be logically coherent — steps follow sensible physical order.",
            f"For a '{difficulty}' scenario: trick scenarios need 0-1 steps if task is done/impossible.",
        ]
    )
    judge_passes = sum(1 for r in assessment.results if r.passed)
    checks_passed += judge_passes

    # ── CALCULATE ACCURACY ────────────────────────────────────────────────────
    accuracy = checks_passed / TOTAL_CHECKS  # e.g. 8/9 = 0.888

    # ── CALCULATE FINAL SCORE ─────────────────────────────────────────────────
    diff_weight = DIFFICULTY_WEIGHTS[difficulty]
    sp_bonus = speed_bonus(elapsed)
    vb_bonus = verbosity_bonus(word_count)

    if accuracy == 1.0:
        # Perfect — full weighted score + bonuses
        final_score = diff_weight + sp_bonus + vb_bonus

    elif accuracy >= (8/9):
        # Partial credit zone — bonuses only, no difficulty multiplier
        final_score = sp_bonus + vb_bonus

    else:
        # Too many failures — zero
        final_score = 0.0

    # ── ASSERT USING SCORE ────────────────────────────────────────────────────
    # Normalize to 0-1 range for kbench (max possible = trick(5) + speed(0.3) + verbosity(0.2) = 5.5)
    MAX_POSSIBLE = 5.5
    normalized = final_score / MAX_POSSIBLE

    kbench.assertions.assert_true(
        normalized > 0,
        expectation=(
            f"[id={id}] difficulty={difficulty} | "
            f"checks={checks_passed}/{TOTAL_CHECKS} | "
            f"accuracy={accuracy:.1%} | "
            f"speed={elapsed:.1f}s(+{sp_bonus}) | "
            f"words={word_count}(+{vb_bonus}) | "
            f"steps={step_count}/{max_steps} | "
            f"score={final_score:.2f}/{MAX_POSSIBLE}"
        )
    )

    return normalized

# ── RUN ───────────────────────────────────────────────────────────────────────
robot_replan.evaluate(evaluation_data=df, llm=[kbench.llm])

