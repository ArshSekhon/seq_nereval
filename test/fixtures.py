from seqnereval.models import Span
import random
from typing import List
from seqnereval.models import ScoreCard

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


def generate_scorecard_fixture():
    scorecard = ScoreCard()
    scorecard.correct = generate_random_gold_pred_span_pairs(5)
    scorecard.incorrect = generate_random_gold_pred_span_pairs(5)
    scorecard.partial = generate_random_gold_pred_span_pairs(5)
    scorecard.missed = generate_random_span_list('missed', random.randint(0, 5))
    scorecard.spurious = generate_random_span_list('missed', random.randint(0, 5))
    scorecard.possible = random.randint(0,100)
    scorecard.actual = random.randint(0,100)
    scorecard.precision = random.random()
    scorecard.recall = random.random()
    scorecard.f1 = random.random()

    return scorecard
