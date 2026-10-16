from types import SimpleNamespace

from langgraph.pregel import Pregel

from travel_agent.graph import check_budget, graph


def test_graph_compiles() -> None:
    assert isinstance(graph, Pregel)


def test_check_budget_uses_state() -> None:
    runtime = SimpleNamespace(state={"budget_eur": 500})
    assert check_budget.func(cost_per_day_eur=100, days=4, runtime=runtime).startswith("OK")
    assert check_budget.func(cost_per_day_eur=100, days=6, runtime=runtime).startswith("Over budget")


def test_check_budget_without_budget() -> None:
    runtime = SimpleNamespace(state={})
    assert check_budget.func(cost_per_day_eur=100, days=4, runtime=runtime).startswith("No budget")
