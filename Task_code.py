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
