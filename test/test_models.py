from seqnereval.models import NEREntitySpan, NERResultAggregator
import random
from typing import List
from pytest_mock import MockerFixture
from .fixtures import generate_random_gold_pred_span_pairs, generate_error_scheme_fixture, generate_random_entity_span

def test_NEREntitySpan_spans_same_tokens_as():
    entity = NEREntitySpan('test', 10, 15)

    same_span_entity = NEREntitySpan('test', 10, 15)
    assert entity.spans_same_tokens_as(same_span_entity) == True

    entity_contained_in_span = NEREntitySpan('test', 11, 12)
    assert entity.spans_same_tokens_as(entity_contained_in_span) == False

    entity_overlap_span_left = NEREntitySpan('test', 7, 10)
    assert entity.spans_same_tokens_as(entity_overlap_span_left) == False

    entity_overlap_span_right = NEREntitySpan('test', 15, 20)
    assert entity.spans_same_tokens_as(entity_overlap_span_right) == False

    entity_non_overlap_left = NEREntitySpan('test', 0, 7)
    assert entity.spans_same_tokens_as(entity_non_overlap_left) == False

    entity_non_overlap_right = NEREntitySpan('test', 20, 27)
    assert entity.spans_same_tokens_as(entity_non_overlap_right) == False

def test_NEREntitySpan_overlaps_with():
    entity = NEREntitySpan('test', 10, 15)

    same_span_entity = NEREntitySpan('test', 10, 15)
    assert entity.overlaps_with(same_span_entity) == True

    entity_contained_in_span = NEREntitySpan('test', 11, 12)
    assert entity.overlaps_with(entity_contained_in_span) == True

    entity_overlap_span_left = NEREntitySpan('test', 7, 10)
    assert entity.overlaps_with(entity_overlap_span_left) == True

    entity_overlap_span_right = NEREntitySpan('test', 15, 20)
    assert entity.overlaps_with(entity_overlap_span_right) == True

    entity_non_overlap_left = NEREntitySpan('test', 0, 7)
    assert entity.overlaps_with(entity_non_overlap_left) == False

    entity_non_overlap_right = NEREntitySpan('test', 20, 27)
    assert entity.overlaps_with(entity_non_overlap_right) == False

def test_NEREntitySpan__str__():
    assert NEREntitySpan('test',1,2,['X1','X2']).__str__() == "(Entity Type: \"test\", Token Span IDX:(1, 2), Tokens:['X1', 'X2'], Context:['X1', 'X2'])"

def test_NEREntitySpan__repr__():
    assert NEREntitySpan('test',1,2,['X1','X2']).__repr__() == "(Entity Type: \"test\", Token Span IDX:(1, 2), Tokens:['X1', 'X2'], Context:['X1', 'X2'])"

def test_NEREntitySpan__hash__():
    assert NEREntitySpan('test',1,2,['X1','X2']) in set([NEREntitySpan('test',1,2,['X1','X2'])])
    assert NEREntitySpan('test',1,3,['X1','X2']) not in set([NEREntitySpan('test',1,2,['X1','X2'])])

def test_NERResultAggregator_append_results(mocker: MockerFixture):
    empty_results = NERResultAggregator()
    spy = mocker.spy(empty_results,'refresh_metrics')
    results_to_append = NERResultAggregator()

    """
    TEST #1: check if the merger of error scenarios work.
    """

    gold_pred_span_fixtures_for_scenarios = {scheme: generate_random_gold_pred_span_pairs(5)
                                             for scheme in ['type_match_span_match', 'unecessary_predicted_entity', 'missed_gold_entity',
                                                            'type_mismatch_span_match', 'type_match_span_partial', 'type_mismatch_span_partial']}

    results_to_append.type_match_span_match = gold_pred_span_fixtures_for_scenarios[
        'type_match_span_match']
    results_to_append.unecessary_predicted_entity = gold_pred_span_fixtures_for_scenarios[
        'unecessary_predicted_entity']
    results_to_append.missed_gold_entity = gold_pred_span_fixtures_for_scenarios[
        'missed_gold_entity']
    results_to_append.type_mismatch_span_match = gold_pred_span_fixtures_for_scenarios[
        'type_mismatch_span_match']
    results_to_append.type_match_span_partial = gold_pred_span_fixtures_for_scenarios[
        'type_match_span_partial']
    results_to_append.type_mismatch_span_partial = gold_pred_span_fixtures_for_scenarios[
        'type_mismatch_span_partial']

    results_to_append.strict_match = generate_error_scheme_fixture()
    results_to_append.type_match = generate_error_scheme_fixture()
    results_to_append.partial_match = generate_error_scheme_fixture()
    results_to_append.bounds_match = generate_error_scheme_fixture()

    empty_results.append_result_aggregator(results_to_append)

    for errorScenario, key in zip([empty_results.type_match_span_match, empty_results.unecessary_predicted_entity,
                                   empty_results.missed_gold_entity, empty_results.type_mismatch_span_match, empty_results.type_match_span_partial,
                                   empty_results.type_mismatch_span_partial],
                                  ['type_match_span_match', 'unecessary_predicted_entity', 'missed_gold_entity',
                                   'type_mismatch_span_match', 'type_match_span_partial', 'type_mismatch_span_partial']
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

    another_results_to_append = NERResultAggregator()
    another_results_to_append.strict_match = generate_error_scheme_fixture()
    another_results_to_append.type_match = generate_error_scheme_fixture()
    another_results_to_append.partial_match = generate_error_scheme_fixture()
    another_results_to_append.bounds_match = generate_error_scheme_fixture()

    empty_results.append_result_aggregator(another_results_to_append)

    # check if the strict_match result schemes were merged successfully
    for key in empty_results.strict_match.keys():
        if type(empty_results.strict_match[key]) is list:
            assert empty_results.strict_match[key] == (results_to_append.strict_match[key]
                                                       + another_results_to_append.strict_match[key])
    # check if the type_match result schemes were merged successfully
    for key in empty_results.type_match.keys():
        if type(empty_results.type_match[key]) is list:
            assert empty_results.type_match[key] == (results_to_append.type_match[key]
                                                       + another_results_to_append.type_match[key])
    # check if the partial_match result schemes were merged successfully
    for key in empty_results.partial_match.keys():
        if type(empty_results.partial_match[key]) is list:
            assert empty_results.partial_match[key] == (results_to_append.partial_match[key]
                                                       + another_results_to_append.partial_match[key])

    # check if the bounds_match result schemes were merged successfully
    for key in empty_results.bounds_match.keys():
        if type(empty_results.bounds_match[key]) is list:
            assert empty_results.bounds_match[key] == (results_to_append.bounds_match[key]
                                                       + another_results_to_append.bounds_match[key])

    assert spy.call_count==2

def test_NERResultAggregator_add_type_match_span_match(mocker: MockerFixture):
    result =  NERResultAggregator()
    spy =  mocker.spy(result,'refresh_metrics')
    gold = generate_random_entity_span('gold')
    pred =  generate_random_entity_span('pred')
    result.add_type_match_span_match(gold, pred)

    assert len(result.type_match_span_match)==1

    assert len(result.strict_match["correct"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.strict_match.values()])==1
    assert len(result.type_match["correct"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert len(result.partial_match["correct"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert len(result.bounds_match["correct"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert spy.call_count==1

def test_NERResultAggregator_add_unecessary_predicted_entity(mocker: MockerFixture):
    result =  NERResultAggregator()
    spy =  mocker.spy(result,'refresh_metrics')
    pred =  generate_random_entity_span('pred')
    result.add_unecessary_predicted_entity(pred)

    assert len(result.unecessary_predicted_entity)==1
    
    assert len(result.strict_match["spurious"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.strict_match.values()])==1
    assert len(result.type_match["spurious"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert len(result.partial_match["spurious"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert len(result.bounds_match["spurious"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert spy.call_count==1

def test_NERResultAggregator_add_missed_gold_entity(mocker: MockerFixture):
    result =  NERResultAggregator()
    spy =  mocker.spy(result,'refresh_metrics')
    gold =  generate_random_entity_span('gold')
    result.add_missed_gold_entity(gold)

    assert len(result.missed_gold_entity)==1
    
    assert len(result.strict_match["missed"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.strict_match.values()])==1
    assert len(result.type_match["missed"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert len(result.partial_match["missed"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert len(result.bounds_match["missed"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert spy.call_count==1

def test_NERResultAggregator_add_type_mismatch_span_match(mocker: MockerFixture):
    result =  NERResultAggregator()
    spy =  mocker.spy(result,'refresh_metrics')

    gold = generate_random_entity_span('gold')
    pred =  generate_random_entity_span('pred')
    result.add_type_mismatch_span_match(gold, pred)

    assert len(result.type_mismatch_span_match)==1
    assert len(result.strict_match["incorrect"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.strict_match.values()])==1
    assert len(result.type_match["incorrect"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert len(result.partial_match["correct"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert len(result.bounds_match["correct"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert spy.call_count==1

def test_NERResultAggregator_add_type_mismatch_span_partial(mocker: MockerFixture):
    result =  NERResultAggregator()
    spy =  mocker.spy(result,'refresh_metrics')

    gold = generate_random_entity_span('gold')
    pred =  generate_random_entity_span('pred')
    result.add_type_mismatch_span_partial(gold, pred)

    assert len(result.type_mismatch_span_partial)==1
    assert len(result.strict_match["incorrect"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.strict_match.values()])==1
    assert len(result.type_match["incorrect"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert len(result.partial_match["partial"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert len(result.bounds_match["incorrect"])==1 and sum([len(agg) if type(agg) is list else 0
                                                                    for agg in result.type_match.values()])==1
    assert spy.call_count==1
