"""Travel planning agent that keeps user preferences in the agent state."""

from __future__ import annotations

import os
from typing import Literal

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

model = init_chat_model(
    model="google/gemini-3.5-flash-lite",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

TravelStyle = Literal["adventure", "relaxation", "culture"]


# Must inherit from AgentState to keep the "messages" field
class State(AgentState):
    budget_eur: float | None
    travel_style: TravelStyle | None
    departure_city: str | None


@tool
def read_preferences(runtime: ToolRuntime[None, State]) -> dict:
    """Read the traveler's preferences (budget, travel style, departure city)."""
    return {
        "budget_eur": runtime.state.get("budget_eur"),
        "travel_style": runtime.state.get("travel_style"),
        "departure_city": runtime.state.get("departure_city"),
    }


@tool
def update_preferences(
    runtime: ToolRuntime[None, State],
    budget_eur: float | None = None,
    travel_style: TravelStyle | None = None,
    departure_city: str | None = None,
) -> Command:
    """Update the traveler's preferences. Only the provided preferences are updated.

    Args:
        budget_eur: Total budget for the whole trip (not per day).
        travel_style: Kind of trip the traveler prefers.
        departure_city: City the traveler leaves from.
    """
    update = {}
    if budget_eur is not None:
        update["budget_eur"] = budget_eur
    if travel_style is not None:
        update["travel_style"] = travel_style
    if departure_city is not None:
        update["departure_city"] = departure_city
    return Command(
        update={
            **update,
            "messages": [
                ToolMessage("Preferences updated", tool_call_id=runtime.tool_call_id),
            ],
        }
    )


@tool
def check_budget(cost_per_day_eur: float, days: int, runtime: ToolRuntime[None, State]) -> str:
    """Check whether a trip of `days` days at `cost_per_day_eur` fits the traveler's budget."""
    budget = runtime.state.get("budget_eur")
    if budget is None:
        return "No budget set: ask the traveler for one."
    total = cost_per_day_eur * days
    if total <= budget:
        return f"OK: {total:.0f} EUR fits the {budget:.0f} EUR budget ({budget - total:.0f} EUR left)."
    return f"Over budget: {total:.0f} EUR exceeds the {budget:.0f} EUR budget by {total - budget:.0f} EUR."


graph = create_agent(
    model=model,
    tools=[read_preferences, update_preferences, check_budget],
    state_schema=State,
    system_prompt=(
        "You are a travel planning assistant. "
        "Start by reading the traveler's preferences, ask for any missing one, "
        "and save a preference only when the traveler explicitly changes it. "
        "Before writing any plan, call check_budget with your estimated cost per day. "
        "Only once the budget is validated, write your final answer: "
        "the full day-by-day plan, followed by a one-line budget summary."
    ),
    name="travel_agent",
)
