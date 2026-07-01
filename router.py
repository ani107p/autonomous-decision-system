"""
Benchmark task set: multi-step decision/reasoning problems with
programmatically computable ground truth, so pipeline accuracy can be
measured exactly rather than estimated.

Each task models a lightweight "business decision" scenario (budget
allocation, ROI comparison, risk-adjusted scoring) -- the kind of
structured-but-nontrivial arithmetic reasoning an autonomous decision
agent is actually used for.
"""
from __future__ import annotations
import random
from dataclasses import dataclass


@dataclass
class DecisionTask:
    id: str
    description: str
    ground_truth: float
    context: str


def _roi_task(i: int, rng: random.Random) -> DecisionTask:
    investment = rng.choice([10000, 25000, 50000, 100000, 200000])
    revenue = investment * rng.uniform(1.1, 2.4)
    cost_of_capital = rng.uniform(0.05, 0.15)
    roi = ((revenue - investment) / investment) - cost_of_capital
    desc = (
        f"A project requires an initial investment of ${investment:,.0f} and is "
        f"projected to generate ${revenue:,.0f} in returns. The cost of capital "
        f"is {cost_of_capital*100:.1f}%. What is the risk-adjusted ROI (as a "
        f"decimal fraction, i.e. (revenue - investment)/investment - cost_of_capital)?"
    )
    return DecisionTask(id=f"TASK-ROI-{i:03d}", description=desc, ground_truth=round(roi, 4),
                         context=f"investment={investment}, revenue={revenue:.2f}, coc={cost_of_capital:.4f}")


def _budget_allocation_task(i: int, rng: random.Random) -> DecisionTask:
    total_budget = rng.choice([50000, 100000, 250000, 500000])
    dept_a_pct = rng.uniform(0.25, 0.45)
    dept_b_pct = rng.uniform(0.20, 0.35)
    dept_c_pct = 1 - dept_a_pct - dept_b_pct
    dept_c_alloc = total_budget * dept_c_pct
    desc = (
        f"A total budget of ${total_budget:,.0f} is split across three departments. "
        f"Department A gets {dept_a_pct*100:.1f}%, Department B gets {dept_b_pct*100:.1f}%, "
        f"and Department C gets the remainder. How much (in dollars) does Department C receive?"
    )
    return DecisionTask(id=f"TASK-BUDGET-{i:03d}", description=desc, ground_truth=round(dept_c_alloc, 2),
                         context=f"total={total_budget}, c_pct={dept_c_pct:.4f}")


def _vendor_score_task(i: int, rng: random.Random) -> DecisionTask:
    price_score = rng.uniform(4, 10)
    quality_score = rng.uniform(4, 10)
    reliability_score = rng.uniform(4, 10)
    weights = (0.4, 0.35, 0.25)
    weighted = price_score * weights[0] + quality_score * weights[1] + reliability_score * weights[2]
    desc = (
        f"A vendor scores {price_score:.1f}/10 on price, {quality_score:.1f}/10 on quality, "
        f"and {reliability_score:.1f}/10 on reliability. Using weights of "
        f"{weights[0]} (price), {weights[1]} (quality), {weights[2]} (reliability), "
        f"what is the weighted composite score?"
    )
    return DecisionTask(id=f"TASK-VENDOR-{i:03d}", description=desc, ground_truth=round(weighted, 3),
                         context=f"weights={weights}")


def generate_tasks(n: int = 30, seed: int = 7) -> list[DecisionTask]:
    rng = random.Random(seed)
    generators = [_roi_task, _budget_allocation_task, _vendor_score_task]
    tasks = []
    for i in range(n):
        gen = generators[i % len(generators)]
        tasks.append(gen(i, rng))
    return tasks
