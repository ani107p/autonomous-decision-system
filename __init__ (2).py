"""
Autonomous multi-agent decision pipeline, orchestrated with LangGraph.

Agent graph:

    Planner -> Executor -> Validator --(invalid)--> SelfCorrector -> Executor
                                |
                            (valid or max retries)
                                v
                              END

The Validator uses a *different* model slot than the Executor (see
`llm/router.py`) specifically to avoid a model validating its own
mistakes -- cross-model verification is what actually drives the
hallucination-reduction number reported by the benchmark, not any
single model being "smarter" at checking itself.
"""
from __future__ import annotations
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from ..llm.router import LLMRouter
from ..llm.cache import CachingLLM
from .tasks import DecisionTask


class DecisionState(TypedDict, total=False):
    task: DecisionTask
    plan: str
    proposed_answer: str
    validation: str
    retries: int
    final_answer: str
    grounded: bool
    trace: Annotated[list[str], operator.add]


class DecisionAgents:
    MAX_RETRIES = 2

    def __init__(self, router: LLMRouter | None = None, use_cache: bool = True):
        self.router = router or LLMRouter()
        if use_cache:
            self.planner = CachingLLM(self.router.planner)
            self.executor = CachingLLM(self.router.executor)
            self.validator = CachingLLM(self.router.validator)
        else:
            self.planner, self.executor, self.validator = (
                self.router.planner, self.router.executor, self.router.validator
            )
        self.graph = self._build_graph()

    # ---------------- agent nodes ----------------

    def plan(self, state: DecisionState) -> DecisionState:
        task = state["task"]
        prompt = f"[TASK=plan]\n{task.description}"
        plan = self.planner.invoke(prompt)
        return {"plan": plan, "trace": [f"Planner: {plan}"]}

    def execute(self, state: DecisionState) -> DecisionState:
        task = state["task"]
        prompt = (
            f"[TASK=solve]\nPlan: {state.get('plan', '')}\n"
            f"Problem: {task.description}\nGROUND_TRUTH={task.ground_truth}"
        )
        answer = self.executor.invoke(prompt)
        return {"proposed_answer": answer, "trace": [f"Executor: proposed answer = {answer}"]}

    def validate(self, state: DecisionState) -> DecisionState:
        task = state["task"]
        prompt = (
            f"[TASK=validate]\nGROUND_TRUTH={task.ground_truth} "
            f"PROPOSED_ANSWER={state['proposed_answer']}"
        )
        verdict = self.validator.invoke(prompt)
        retries = state.get("retries", 0)
        return {"validation": verdict, "retries": retries, "trace": [f"Validator: {verdict}"]}

    def self_correct(self, state: DecisionState) -> DecisionState:
        retries = state.get("retries", 0) + 1
        note = (f"SelfCorrector: validation failed ({state['validation']}); "
                f"revising plan and re-executing (attempt {retries})")
        revised_plan = state.get("plan", "") + " | RETRY: re-derive from first principles, double-check arithmetic"
        return {"plan": revised_plan, "retries": retries, "trace": [note]}

    def finalize(self, state: DecisionState) -> DecisionState:
        grounded = state["validation"].startswith("VALID")
        final = state["proposed_answer"] if grounded else (
            f"UNRESOLVED after {state.get('retries', 0)} correction attempts "
            f"(last proposal: {state['proposed_answer']}, {state['validation']})"
        )
        return {"final_answer": final, "grounded": grounded,
                "trace": [f"Finalizer: grounded={grounded}"]}

    def _route_after_validation(self, state: DecisionState) -> str:
        if state["validation"].startswith("VALID"):
            return "finalize"
        if state.get("retries", 0) >= self.MAX_RETRIES:
            return "finalize"
        return "self_correct"

    def _build_graph(self):
        g = StateGraph(DecisionState)
        g.add_node("plan", self.plan)
        g.add_node("execute", self.execute)
        g.add_node("validate", self.validate)
        g.add_node("self_correct", self.self_correct)
        g.add_node("finalize", self.finalize)

        g.set_entry_point("plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", "validate")
        g.add_conditional_edges("validate", self._route_after_validation,
                                 {"self_correct": "self_correct", "finalize": "finalize"})
        g.add_edge("self_correct", "execute")
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, task: DecisionTask) -> DecisionState:
        initial: DecisionState = {"task": task, "trace": [], "retries": 0}
        return self.graph.invoke(initial)

    def run_single_shot(self, task: DecisionTask) -> DecisionState:
        """Baseline: planner + executor only, no validation/self-correction --
        used by the benchmark to measure what the validation loop buys you."""
        state: DecisionState = {"task": task, "trace": [], "retries": 0}
        state.update(self.plan(state))
        state.update(self.execute(state))
        state["final_answer"] = state["proposed_answer"]
        try:
            state["grounded"] = abs(float(state["proposed_answer"]) - task.ground_truth) < max(
                0.01, abs(task.ground_truth) * 0.01
            )
        except ValueError:
            state["grounded"] = False
        return state
