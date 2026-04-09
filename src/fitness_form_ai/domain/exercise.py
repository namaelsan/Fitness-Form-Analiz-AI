from __future__ import annotations

from abc import ABC

from fitness_form_ai.domain.rep_frame import RepFrame
from fitness_form_ai.domain.rules import Rule, RuleContext


class Exercise(ABC):
    def __init__(
        self,
        name: str,
        rules: list[Rule],
        primary_joints: list[str],
        start_phase: str = "concentric",
    ) -> None:
        self.name = name
        self.rules = rules
        self.primary_joints = primary_joints
        self.start_phase = start_phase

    def apply_rules(self, rep_data: list[RepFrame]) -> tuple[bool, str]:
        failed_reasons = [rule.rule_name for rule in self.rules if not rule.apply(rep_data)]
        if failed_reasons:
            return False, f"Failed: {', '.join(failed_reasons)}"
        return True, ""

    def get_rule_states(self, context: RuleContext) -> list[tuple[str, str]]:
        return [(rule.rule_name, rule.describe_current(context)) for rule in self.rules]
