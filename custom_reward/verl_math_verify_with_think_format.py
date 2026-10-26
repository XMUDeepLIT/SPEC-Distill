import re
import logging
from contextlib import contextmanager

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from latex2sympy2_extended.latex2sympy2 import NormalizationConfig


@contextmanager
def silence_logging(logger_name: str = "math_verify", level: int = logging.ERROR):
    logger = logging.getLogger(logger_name)
    prev_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(prev_level)


def think_format_reward(solution_str: str) -> float:
    """Check if the response follows <think>...</think> format."""
    pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    match = pattern.search(solution_str)
    if not match:
        return 0.0

    think_start = solution_str.find("<think>")
    think_end = solution_str.find("</think>")
    if think_start != 0:
        return 0.0
    if solution_str.count("<think>") != 1 or solution_str.count("</think>") != 1:
        return 0.0
    if think_end <= think_start:
        return 0.0

    return 1.0


def acc_reward(solution_str: str, ground_truth: str) -> float:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(
            LatexExtractionConfig(
                boxed_match_priority=0,
                normalization_config=NormalizationConfig(
                    basic_latex=True,
                    units=True,
                    malformed_operators=False,
                    nits=False,
                    boxed="all",
                    equations=False,
                ),
            ),
        ),
    )
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    with silence_logging("math_verify", logging.ERROR):
        try:
            ret_score, _ = verify_func([ground_truth_boxed], [solution_str])
        except Exception:
            pass
        except TimeoutException:
            ret_score = 0.0
    return ret_score


def compute_score(
    solution_str: str,
    ground_truth: str,
    data_source=None,
    extra_info=None,
    format_weight: float = 0.1,
) -> dict:
    acc = acc_reward(solution_str, ground_truth)
    fmt = think_format_reward(solution_str)
    score = (1.0 - format_weight) * acc + format_weight * fmt
    return {
        "score": score,
        "acc": acc,
        "think_format": fmt,
    }
