from seqnereval.models import Span, ResultAggregator
import random
from typing import List
from pytest_mock import MockerFixture
from ..fixtures import generate_random_gold_pred_span_pairs, generate_scorecard_fixture, generate_random_span

def test_ResultAggregator_append_results(mocker: MockerFixture):
    empty_results = ResultAggregator()
    spy = mocker.spy(empty_results,'recalculate_metrics_for_all_scorecards')
    results_to_append = ResultAggregator()

    """
    TEST #1: check if the merger of error scenarios work.
    """

    gold_pred_span_fixtures_for_scenarios = {scheme: generate_random_gold_pred_span_pairs(5)
                                             for scheme in ['type_match_bounds_match', 'unecessary_predicted_span', 'missed_gold_span',
                                                            'type_mismatch_bounds_match', 'type_match_bounds_partial', 'type_mismatch_bounds_partial']}

    results_to_append.type_match_bounds_match = gold_pred_span_fixtures_for_scenarios['type_match_bounds_match']
    results_to_append.unecessary_predicted_span = gold_pred_span_fixtures_for_scenarios['unecessary_predicted_span']
    results_to_append.missed_gold_span = gold_pred_span_fixtures_for_scenarios['missed_gold_span']
    results_to_append.type_mismatch_bounds_match = gold_pred_span_fixtures_for_scenarios['type_mismatch_bounds_match']
    results_to_append.type_match_bounds_partial = gold_pred_span_fixtures_for_scenarios['type_match_bounds_partial']
    results_to_append.type_mismatch_bounds_partial = gold_pred_span_fixtures_for_scenarios['type_mismatch_bounds_partial']

    results_to_append.strict_match = generate_scorecard_fixture()
    results_to_append.type_match = generate_scorecard_fixture()
    results_to_append.partial_match = generate_scorecard_fixture()
    results_to_append.bounds_match = generate_scorecard_fixture()

    empty_results.append_result_aggregator(results_to_append)

    for errorScenario, key in zip([empty_results.type_match_bounds_match, empty_results.unecessary_predicted_span,
                                   empty_results.missed_gold_span, empty_results.type_mismatch_bounds_match, empty_results.type_match_bounds_partial,
                                   empty_results.type_mismatch_bounds_partial],
                                  ['type_match_bounds_match', 'unecessary_predicted_span', 'missed_gold_span',
                                   'type_mismatch_bounds_match', 'type_match_bounds_partial', 'type_mismatch_bounds_partial']
                                  ):
        # assert error scenarios

        #print(errorScenario, "\n", gold_pred_span_fixtures_for_scenarios[key],"\n\n")
        assert len(errorScenario) == len(
            gold_pred_span_fixtures_for_scenarios[key])

        for idx, (gold, pred) in enumerate(errorScenario):
            assert (
                gold, pred) == gold_pred_span_fixtures_for_scenarios[key][idx]

    """
    TEST #2: Check if merger of result schemes work
    """

    another_results_to_append = ResultAggregator()
    another_results_to_append.strict_match = generate_scorecard_fixture()
    another_results_to_append.type_match = generate_scorecard_fixture()
    another_results_to_append.partial_match = generate_scorecard_fixture()
    another_results_to_append.bounds_match = generate_scorecard_fixture()

    empty_results.append_result_aggregator(another_results_to_append)

    # check if the strict_match result schemes were merged successfully
    for key in empty_results.strict_match.__dict__.keys():
        if type(empty_results.strict_match.__dict__[key]) is list:
            assert empty_results.strict_match.__dict__[key] == (results_to_append.strict_match.__dict__[key]
                                                       + another_results_to_append.strict_match.__dict__[key])
    # check if the type_match result schemes were merged successfully
    for key in empty_results.type_match.__dict__.keys():
        if type(empty_results.type_match.__dict__[key]) is list:
            assert empty_results.type_match.__dict__[key] == (results_to_append.type_match.__dict__[key]
                                                       + another_results_to_append.type_match.__dict__[key])
    # check if the partial_match result schemes were merged successfully
    for key in empty_results.partial_match.__dict__.keys():
        if type(empty_results.partial_match.__dict__[key]) is list:
            assert empty_results.partial_match.__dict__[key] == (results_to_append.partial_match.__dict__[key]
                                                       + another_results_to_append.partial_match.__dict__[key])

    # check if the bounds_match result schemes were merged successfully
    for key in empty_results.bounds_match.__dict__.keys():
        if type(empty_results.bounds_match.__dict__[key]) is list:
            assert empty_results.bounds_match.__dict__[key] == (results_to_append.bounds_match.__dict__[key]
                                                       + another_results_to_append.bounds_match.__dict__[key])

    assert spy.call_count==2

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
