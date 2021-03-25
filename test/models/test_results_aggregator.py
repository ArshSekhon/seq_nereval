from copy import copy
import random
from typing import List
from pytest_mock import MockerFixture
from unittest.mock import MagicMock

from seqnereval.models import Span, ResultAggregator

from ..fixtures import (
    generate_random_scorecard_fixture, 
    generate_random_results_aggregator,
    generate_random_gold_pred_span_pairs, 
    generate_random_span
)

def test_ResultAggregator_append_results(mocker: MockerFixture):
    # prepare res
    res = generate_random_results_aggregator()
    
    res.strict_match.appendScoreCard = MagicMock()
    res.type_match.appendScoreCard = MagicMock()
    res.partial_match.appendScoreCard = MagicMock()
    res.bounds_match.appendScoreCard = MagicMock()

    initial_type_match_bounds_match = copy(res.type_match_bounds_match)
    initial_unecessary_predicted_span = copy(res.unecessary_predicted_span)
    initial_missed_gold_span = copy(res.missed_gold_span)
    initial_type_mismatch_bounds_match = copy(res.type_mismatch_bounds_match)
    initial_type_match_bounds_partial = copy(res.type_match_bounds_partial)
    initial_type_mismatch_bounds_partial = copy(res.type_mismatch_bounds_partial)

    spy = mocker.spy(res,'recalculate_metrics_for_all_scorecards')

    res_to_append = generate_random_results_aggregator()

    res.append_result_aggregator(res_to_append)

    # TEST 1: check if the lists of error scenarios were merged correctly
    assert res.type_match_bounds_match == initial_type_match_bounds_match + res_to_append.type_match_bounds_match
    assert res.unecessary_predicted_span == initial_unecessary_predicted_span + res_to_append.unecessary_predicted_span
    assert res.missed_gold_span == initial_missed_gold_span + res_to_append.missed_gold_span
    assert res.type_mismatch_bounds_match == initial_type_mismatch_bounds_match + res_to_append.type_mismatch_bounds_match
    assert res.type_match_bounds_partial == initial_type_match_bounds_partial + res_to_append.type_match_bounds_partial
    assert res.type_mismatch_bounds_partial == initial_type_mismatch_bounds_partial + res_to_append.type_mismatch_bounds_partial

    # Test 2: Check if scorecards were merged.
    res.strict_match.appendScoreCard.assert_called_with(res_to_append.strict_match)
    res.type_match.appendScoreCard.assert_called_with(res_to_append.type_match)
    res.partial_match.appendScoreCard.assert_called_with(res_to_append.partial_match)
    res.bounds_match.appendScoreCard.assert_called_with(res_to_append.bounds_match)
    
    assert spy.call_count==1


def test_ResultAggregator_add_type_match_bounds_match(mocker: MockerFixture):
    result =  ResultAggregator()
    spy =  mocker.spy(result,'recalculate_metrics_for_all_scorecards')
    gold = generate_random_span('gold')
    pred =  generate_random_span('pred')
    result.add_type_match_bounds_match(gold, pred)

    assert len(result.type_match_bounds_match)==1

    assert len(result.strict_match.correct)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.strict_match.__dict__.values()])==1
    assert len(result.type_match.correct)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.partial_match.correct)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.bounds_match.correct)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert spy.call_count==1

def test_ResultAggregator_add_unecessary_predicted_span(mocker: MockerFixture):
    result =  ResultAggregator()
    spy =  mocker.spy(result,'recalculate_metrics_for_all_scorecards')
    pred =  generate_random_span('pred')
    result.add_unecessary_predicted_span(pred)

    assert len(result.unecessary_predicted_span)==1
    
    assert len(result.strict_match.spurious)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.strict_match.__dict__.values()])==1
    assert len(result.type_match.spurious)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.partial_match.spurious)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.bounds_match.spurious)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert spy.call_count==1

def test_ResultAggregator_add_missed_gold_span(mocker: MockerFixture):
    result =  ResultAggregator()
    spy =  mocker.spy(result,'recalculate_metrics_for_all_scorecards')
    gold =  generate_random_span('gold')
    result.add_missed_gold_span(gold)

    assert len(result.missed_gold_span)==1
    
    assert len(result.strict_match.missed)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.strict_match.__dict__.values()])==1
    assert len(result.type_match.missed)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.partial_match.missed)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.bounds_match.missed)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert spy.call_count==1

def test_ResultAggregator_add_type_mismatch_bounds_match(mocker: MockerFixture):
    result =  ResultAggregator()
    spy =  mocker.spy(result,'recalculate_metrics_for_all_scorecards')

    gold = generate_random_span('gold')
    pred =  generate_random_span('pred')
    result.add_type_mismatch_bounds_match(gold, pred)

    assert len(result.type_mismatch_bounds_match)==1
    assert len(result.strict_match.incorrect)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.strict_match.__dict__.values()])==1
    assert len(result.type_match.incorrect)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.partial_match.correct)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.bounds_match.correct)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert spy.call_count==1

def test_ResultAggregator_add_type_match_bounds_partial(mocker: MockerFixture):
    result =  ResultAggregator()
    spy =  mocker.spy(result,'recalculate_metrics_for_all_scorecards')

    gold = generate_random_span('gold')
    pred =  generate_random_span('pred')
    result.add_type_match_bounds_partial(gold, pred)

    assert len(result.type_match_bounds_partial)==1
    assert len(result.strict_match.incorrect)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.strict_match.__dict__.values()])==1
    assert len(result.type_match.correct)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.partial_match.partial)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.bounds_match.incorrect)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert spy.call_count==1

def test_ResultAggregator_add_type_mismatch_bounds_partial(mocker: MockerFixture):
    result =  ResultAggregator()
    spy =  mocker.spy(result,'recalculate_metrics_for_all_scorecards')

    gold = generate_random_span('gold')
    pred =  generate_random_span('pred')
    result.add_type_mismatch_bounds_partial(gold, pred)

    assert len(result.type_mismatch_bounds_partial)==1
    assert len(result.strict_match.incorrect)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.strict_match.__dict__.values()])==1
    assert len(result.type_match.incorrect)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.partial_match.partial)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert len(result.bounds_match.incorrect)==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.__dict__.values()])==1
    assert spy.call_count==1



def test_ResultAggregator_summarize_results():
    results = generate_random_results_aggregator()
    assert results.summarize_result() == {
            "strict_match": results.strict_match.get_summary(),
            "type_match": results.type_match.get_summary(),
            "partial_match": results.partial_match.get_summary(),
            "bounds_match": results.bounds_match.get_summary(),
            "type_match_bounds_match": len(results.type_match_bounds_match),
            "unecessary_predicted_span": len(results.unecessary_predicted_span),
            "missed_gold_span": len(results.missed_gold_span),
            "type_mismatch_bounds_match": len(results.type_mismatch_bounds_match),
            "type_match_bounds_partial": len(results.type_match_bounds_partial),
            "type_mismatch_bounds_partial": len(results.type_mismatch_bounds_partial)
        }