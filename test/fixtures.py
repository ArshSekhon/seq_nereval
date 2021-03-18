from seqnereval.models import Span
import random
from typing import List

def generate_random_span(token_placeholder: str):
    return Span(token_placeholder, random.randint(
            1, 100), random.randint(1, 100))

def generate_random_span_list(token_placeholder: str, count: int) -> List:
    return [
        Span(token_placeholder, random.randint(
            1, 100), random.randint(1, 100))
        for _ in range(count)
    ]


def generate_random_gold_pred_span_pairs(maxCount: int):
    random_length = random.randint(0, maxCount)
    return list(zip(generate_random_span_list('gold', random_length), generate_random_span_list('pred', random_length)))


def generate_error_scheme_fixture():
    return {
        "correct": generate_random_gold_pred_span_pairs(5),
        "incorrect": generate_random_gold_pred_span_pairs(5),
        "partial": generate_random_gold_pred_span_pairs(5),
        "missed": generate_random_span_list('missed', random.randint(0, 5)),
        "spurious": generate_random_gold_pred_span_pairs(5),
        "possible": generate_random_gold_pred_span_pairs(5),
        "actual": generate_random_gold_pred_span_pairs(5),
        "precision": random.random(),
        "recall": random.random(),
        "f1": random.random(),
    }
