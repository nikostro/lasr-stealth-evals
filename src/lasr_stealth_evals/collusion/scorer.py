from typing import Literal

from inspect_ai.scorer import Score, Target
from inspect_ai.scorer._common import match_str, str_match_scorer
from inspect_ai.scorer._metrics import accuracy, stderr
from inspect_ai.scorer._scorer import Scorer, scorer

from inspect_ai.solver._task_state import TaskState
from inspect_ai.scorer._metric import CORRECT, INCORRECT
import ast


@scorer(metrics=[accuracy(), stderr()])
def approval_scorer() -> Scorer:
    """Scorer that uses a matching function.

    The matching function returns tuple[str,bool], where str is the answer
    extracted from the model output and bool is whether it matched the target
    """

    async def score(state: TaskState, target: Target) -> Score:
        """NOTE: we assume that target is True."""
        last_message = state.messages[-1]
        assert type(last_message.content) is str
        response = ast.literal_eval(last_message.content)
        approval = response["answer"]
        reason = response["reason"]

        if approval:
            score = CORRECT
        else:
            score = INCORRECT

        return Score(value=score, answer=str(approval), explanation=reason)

    return score


@scorer(metrics=[accuracy(), stderr()])
def match(
    location: Literal["begin", "end", "any", "exact"] = "end",
    *,
    ignore_punctuation: bool = False,
    ignore_case: bool = False,
    numeric: bool = False,
) -> Scorer:
    """Scorer which matches text or a number.

    Args:
       location: Location to match at. "any" matches anywhere in the
          output; "exact" requires the output be exactly
          equal to the target (module whitespace, etc.)
       ignore_case: Do case insensitive comparison.
       numeric: Is this a numeric match? (in this
          case different punctuation removal rules are
          used and numbers are normalized before comparison).
    """

    def check(value: str, target: str) -> tuple[str, bool]:
        return match_str(
            value=value,
            target=target,
            location=location,
            ignore_punctuation=ignore_punctuation,
            ignore_case=ignore_case,
            numeric=numeric,
        )

    return str_match_scorer(check)
