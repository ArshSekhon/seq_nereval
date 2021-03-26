from seqnereval.models import Span, GoldPredictedPair, ScoreCard, ResultAggregator
import random
from typing import List
from seqnereval.models import ScoreCard

# TODO: split fixtures into separate files

def generate_random_span(type_placeholder: str, min_idx:int = 1, max_idx: int = 1000):
    start_idx = random.randint(min_idx, max_idx)
    end_idx =  random.randint(start_idx, max_idx)
    return Span(type_placeholder, start_idx, end_idx)

def generate_non_overlapping_span_list(type_placeholder: str, count: int) -> List:
    return [
        Span(type_placeholder, idx, idx)
        for idx in range(1,count)
    ]


def generate_random_gold_pred_span_pairs(maxCount: int):
    random_length = random.randint(1, maxCount)
    result = []
    for gold, pred in list(zip(
        generate_non_overlapping_span_list('gold', random_length), 
        generate_non_overlapping_span_list('pred', random_length))):

        result.append(GoldPredictedPair(gold, pred))

    return result

def generate_random_scorecard_fixture():
    scorecard = ScoreCard()
    scorecard.correct = generate_random_gold_pred_span_pairs(random.randint(1,20))
    scorecard.incorrect = generate_random_gold_pred_span_pairs(random.randint(1,20))
    scorecard.partial = generate_random_gold_pred_span_pairs(random.randint(1,20))
    scorecard.spurious = generate_random_gold_pred_span_pairs(random.randint(1,20))
    scorecard.missed = generate_random_gold_pred_span_pairs(random.randint(1,20)) 
    scorecard.recalculate_metrics()
    return scorecard

def create_non_random_scorecard_fixture():
    scorecard = ScoreCard()
    scorecard.correct = [
        GoldPredictedPair(gold_span=Span("tc",0,10),predicted_span= Span("tc",0,10)),
        GoldPredictedPair(gold_span=Span("tc",11,12),predicted_span= Span("tc",11,12)),
        GoldPredictedPair(gold_span=Span("tc",13,16),predicted_span= Span("tc",13,16)),
    ]
    scorecard.incorrect = [
        GoldPredictedPair(gold_span=Span("ti0",0,10),predicted_span= Span("ti1",20,21)),
        GoldPredictedPair(gold_span=Span("ti2",11,12),predicted_span= Span("ti3",21,22)),
        GoldPredictedPair(gold_span=Span("ti4",13,16),predicted_span= Span("ti5",23,26)),
    ]
    scorecard.partial = [
        GoldPredictedPair(gold_span=Span("tp0",0,10),predicted_span= Span("tp1",0,10)),
        GoldPredictedPair(gold_span=Span("tp2",11,12),predicted_span= Span("tp3",11,12)),
        GoldPredictedPair(gold_span=Span("tp4",13,16),predicted_span= Span("tp5",13,16)),
    ]
    scorecard.missed = [
        Span("tm0",0,10), 
        Span("tm1",20,21), 
        Span("tm2",30,31)
    ]
    scorecard.spurious = [
        Span("ts0",0,10), 
        Span("ts1",20,21)
    ]
    scorecard.recalculate_metrics()
    return scorecard

def create_non_random_type_or_partial_scorecard_fixture():
    scorecard = ScoreCard(True)
    scorecard.correct = [
        GoldPredictedPair(gold_span=Span("tc",0,10),predicted_span= Span("tc",0,10)),
        GoldPredictedPair(gold_span=Span("tc",11,12),predicted_span= Span("tc",11,12)),
        GoldPredictedPair(gold_span=Span("tc",13,16),predicted_span= Span("tc",13,16)),
    ]
    scorecard.incorrect = [
        GoldPredictedPair(gold_span=Span("ti0",0,10),predicted_span= Span("ti1",20,21)),
        GoldPredictedPair(gold_span=Span("ti2",11,12),predicted_span= Span("ti3",21,22)),
        GoldPredictedPair(gold_span=Span("ti4",13,16),predicted_span= Span("ti5",23,26)),
    ]
    scorecard.partial = [
        GoldPredictedPair(gold_span=Span("tp0",0,10),predicted_span= Span("tp1",0,10)),
        GoldPredictedPair(gold_span=Span("tp2",11,12),predicted_span= Span("tp3",11,12)),
        GoldPredictedPair(gold_span=Span("tp4",13,16),predicted_span= Span("tp5",13,16)),
    ]
    scorecard.missed = [
        Span("tm0",0,10), 
        Span("tm1",20,21), 
        Span("tm2",30,31)
    ]
    scorecard.spurious = [
        Span("ts0",0,10), 
        Span("ts1",20,21)
    ]
    scorecard.recalculate_metrics()
    return scorecard

def generate_random_results_aggregator():
    res = ResultAggregator()
    res.type_match_bounds_match = generate_random_gold_pred_span_pairs(random.randint(1,20))
    res.unecessary_predicted_span = generate_random_gold_pred_span_pairs(random.randint(1,20))
    res.missed_gold_span = generate_random_gold_pred_span_pairs(random.randint(1,20))
    res.type_mismatch_bounds_match = generate_random_gold_pred_span_pairs(random.randint(1,20))
    res.type_match_bounds_partial = generate_random_gold_pred_span_pairs(random.randint(1,20))
    res.type_mismatch_bounds_partial = generate_random_gold_pred_span_pairs(random.randint(1,20))

    res.strict_match = generate_random_scorecard_fixture()
    res.type_match = generate_random_scorecard_fixture()
    res.partial_match = generate_random_scorecard_fixture()
    res.bounds_match = generate_random_scorecard_fixture()
    return res
