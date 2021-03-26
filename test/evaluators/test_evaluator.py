from seqnereval.evaluators import Evaluator
from seqnereval.models import ResultAggregator, Span, GoldPredictedPair

from ..fixtures import generate_non_overlapping_span_list

import pytest
from unittest.mock import patch, call
from copy import deepcopy
from itertools import chain

import random


def test_document_evaluator__init__():
    gold_spans = generate_non_overlapping_span_list('gold', 10)
    pred_spans = generate_non_overlapping_span_list('pred', 10)

    random.shuffle(gold_spans)
    random.shuffle(pred_spans)

    evaluator = Evaluator(gold_spans, pred_spans)

    assert evaluator.gold_spans == sorted(gold_spans, key=lambda span: (span.start_idx, span.end_idx))
    assert evaluator.predicted_spans == sorted(pred_spans, key=lambda span: (span.start_idx, span.end_idx))


def test_document_evaluator__init__overlapping_span_exception():
    gold_spans = generate_non_overlapping_span_list('gold', 10)
    pred_spans = generate_non_overlapping_span_list('pred', 10)

    overlapping_gold_span = gold_spans + gold_spans
    overlapping_pred_span = pred_spans + pred_spans

    random.shuffle(gold_spans)
    random.shuffle(pred_spans)

    with pytest.raises(Exception) as overlapping_span_exception:
        Evaluator(overlapping_gold_span, pred_spans)
    
    assert str(overlapping_span_exception.value) == "Overlapping Gold Spans found: Overlapping spans are not currently supported."

    with pytest.raises(Exception) as overlapping_span_exception:
        Evaluator(gold_spans, overlapping_pred_span)
    
    assert str(overlapping_span_exception.value) == "Overlapping Predicted Spans found: Overlapping spans are not currently supported."
    
def test_document_evaluator_get_results_before_evaluation():
    gold_spans = generate_non_overlapping_span_list('gold', 10)
    pred_spans = generate_non_overlapping_span_list('pred', 10)

    evaluator = Evaluator(gold_spans, pred_spans)

    with pytest.raises(Exception) as get_result_before_eval_exception:
        evaluator.get_result()
    assert str(get_result_before_eval_exception.value)==("Evaluation has not been performed yet. Please call evaluate() before retrieving results.")
    
    with pytest.raises(Exception) as get_result_before_eval_exception:
        evaluator.get_results_grouped_by_tags()
    assert str(get_result_before_eval_exception.value)==("Evaluation has not been performed yet. Please call evaluate() before retrieving results.")

# TODO: refactor to make tests easier to read
@patch.object(ResultAggregator, 'add_type_match_bounds_match')
def test_document_evaluator_type_match_bounds_match(add_type_match_bounds_match_mock):
    gold_spans = [Span('tag1', 1,10), Span('tag2', 21,30), Span('tag3', 31,40), Span('tag4', 41,50)]
    pred_spans = [Span('tag1', 1,10), Span('tag2', 21,30), Span('tag3', 31,40), Span('tag4', 41,50)]

    evaluator = Evaluator(gold_spans, pred_spans)
    evaluator.evaluate()

    # create a list of calls for each pair
    gold_pred_pair_calls = [call(gold, pred) for gold, pred in list(zip(gold_spans, pred_spans))]
    # for each pair a call should be made to add_type_match_bounds_match() for result and result_by_tag each  
    expected_calls = list(chain.from_iterable(list(zip(gold_pred_pair_calls, gold_pred_pair_calls))))
    
    add_type_match_bounds_match_mock.assert_has_calls(expected_calls)

@patch.object(ResultAggregator, 'add_unecessary_predicted_span')
def test_document_evaluator_unecessary_predicted_entity(add_unecessary_predicted_span_mock):
    gold_spans = [Span('tag1', 1,5)]
    pred_spans = [Span('tag1', 12,20), Span('tag2', 21,30), Span('tag3', 31,40), Span('tag4', 41,50)]
       
    evaluator = Evaluator(gold_spans, pred_spans)
    evaluator.evaluate()
    
    # create a list of calls for each pred
    calls_with_pred = [call(pred) for pred in pred_spans]
    # for each pred a call should be made to add_unecessary_predicted_span() for result and result_by_tag each  
    expected_calls = list(chain.from_iterable(list(zip(calls_with_pred, calls_with_pred))))
    add_unecessary_predicted_span_mock.assert_has_calls(expected_calls)

@patch('seqnereval.models.ResultAggregator.add_missed_gold_span')
def test_document_evaluator_missed_entity(add_missed_gold_span_mock):
    gold_spans = [Span('tag1', 12,20), Span('tag2', 21,30), Span('tag3', 31,40), Span('tag4', 41,50)]
    pred_spans = [Span('tag1', 1,4)]
    evaluator = Evaluator(gold_spans, pred_spans)
    evaluator.evaluate()
    
    # create a list of calls for each gold
    calls_with_gold= [call(gold) for gold in gold_spans]
    # for each gold a call should be made to add_missed_gold_span_mock() for result and result_by_tag each  
    expected_calls = list(chain.from_iterable(list(zip(calls_with_gold, calls_with_gold))))

    add_missed_gold_span_mock.assert_has_calls(
        [call(gold) for gold in sorted(gold_spans*2,key=lambda span: (span.start_idx, span.end_idx))]
    )

@patch.object(ResultAggregator, 'add_type_mismatch_bounds_match')
def test_document_evaluator_type_mismatch_bounds_match(add_type_mismatch_bounds_match_mock):
    gold_spans = [Span('tag1', 1,10), Span('tag2', 21,30), Span('tag3', 31,40), Span('tag4', 41,50)]
    pred_spans = [Span('tag3', 1,10), Span('tag1', 21,30), Span('tag2', 31,40), Span('tag1', 41,50)]

    evaluator = Evaluator(gold_spans, pred_spans)
    evaluator.evaluate()
    
    # create a list of calls for each pair
    gold_pred_pair_calls = [call(gold, pred) for gold, pred in list(zip(gold_spans, pred_spans))]
    # for each pair a call should be made to add_type_mismatch_bounds_match_mock() for result and result_by_tag each  
    expected_calls = list(chain.from_iterable(list(zip(gold_pred_pair_calls, gold_pred_pair_calls))))
    add_type_mismatch_bounds_match_mock.assert_has_calls(expected_calls)

@patch.object(ResultAggregator, 'add_type_match_bounds_partial')
def test_document_evaluator_type_match_bounds_partial(add_type_match_bounds_partial_mock):
    gold_spans = [Span('tag1', 1,10), Span('tag2', 21,30), Span('tag3', 31,40), Span('tag4', 41,50), Span('tag5', 71,80)]
    pred_spans = [Span('tag1', 1,15), Span('tag2', 17,30), Span('tag3', 33,38), Span('tag4', 48,55), Span('tag5', 66,88)]

    evaluator = Evaluator(gold_spans, pred_spans)
    evaluator.evaluate()

    # create a list of calls for each pair
    gold_pred_pair_calls = [call(gold, pred) for gold, pred in list(zip(gold_spans, pred_spans))]
    # for each pair a call should be made to add_type_match_bounds_partial_mock() for result and result_by_tag each  
    expected_calls = list(chain.from_iterable(list(zip(gold_pred_pair_calls, gold_pred_pair_calls))))

    add_type_match_bounds_partial_mock.assert_has_calls(expected_calls)

@patch.object(ResultAggregator, 'add_type_mismatch_bounds_partial')
def test_document_evaluator_type_mismatch_bounds_partial(add_type_mismatch_bounds_partial_mock):
    gold_spans = [Span('tag1', 1,10), Span('tag2', 21,30), Span('tag3', 31,40), Span('tag4', 41,50), Span('tag4', 71,80)]
    pred_spans = [Span('tag3', 1,15), Span('tag1', 17,30), Span('tag2', 33,38), Span('tag1', 48,55), Span('tag1', 66,88)]

    evaluator = Evaluator(gold_spans, pred_spans)
    evaluator.evaluate()

    # create a list of calls for each pair
    gold_pred_pair_calls = [call(gold, pred) for gold, pred in list(zip(gold_spans, pred_spans))]
    # for each pair a call should be made to add_type_match_bounds_partial() for result and result_by_tag each  
    expected_calls = list(chain.from_iterable(list(zip(gold_pred_pair_calls, gold_pred_pair_calls))))

    add_type_mismatch_bounds_partial_mock.assert_has_calls(expected_calls)

