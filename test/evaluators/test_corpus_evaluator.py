from seqnereval.evaluators import CorpusEvaluator, DocumentEvaluator
from seqnereval.models import ResultAggregator

from ..fixtures import generate_random_results_aggregator, generate_non_overlapping_span_list
from unittest.mock import patch, call

import pytest
import pprint
import random

def test_corpus_evaluator__init__with_doc_count_mismatch():
    gold_spans = generate_non_overlapping_span_list('gold', 10)
    pred_spans = generate_non_overlapping_span_list('pred', 10)

    with pytest.raises(Exception) as doc_count_mismatch_exception:
        CorpusEvaluator([gold_spans]*10,[pred_spans]*20)

    assert str(doc_count_mismatch_exception.value)==(f'# of documents for which golden tags were provided 10' 
                f'!= # of documents for which predicted tags were provided 20') 


@patch('seqnereval.evaluators.DocumentEvaluator.evaluate')
@patch('seqnereval.models.ResultAggregator.append_result_aggregator')
def test_corpus_evaluator_evaluate_get_results(result_aggegator_mock, doc_evaluate_mock):
    number_of_doc = random.randint(2, 10)

    mocked_results_from_doc_eval = [generate_random_results_aggregator() for _ in range(number_of_doc)]
    mocked_results_by_tag_from_doc_eval = [{'tag': result} for result in mocked_results_from_doc_eval]

    doc_evaluate_mock.side_effect = list(zip(mocked_results_from_doc_eval, mocked_results_by_tag_from_doc_eval))

    gold_spans = generate_non_overlapping_span_list('gold', 10)
    pred_spans = generate_non_overlapping_span_list('pred', 10)

    evaluator = CorpusEvaluator([gold_spans]*number_of_doc,[pred_spans]*number_of_doc)

    #print(doc_evaluate_mock.evaluate)
    evaluator.evaluate()
    # pylint: disable=no-member # It is a mock therefore assert_called_with attribute exists
    evaluator.get_result().append_result_aggregator.has_calls([call(result) for result in mocked_results_from_doc_eval])


@patch('seqnereval.evaluators.DocumentEvaluator.evaluate')
@patch('seqnereval.models.ResultAggregator.append_result_aggregator')
def test_corpus_evaluator_evaluate_get_results_tags(result_aggegator_mock, doc_evaluate_mock):
    number_of_doc = random.randint(2, 10)

    mocked_results_from_doc_eval = [generate_random_results_aggregator() for _ in range(number_of_doc)]
    mocked_results_by_tag_from_doc_eval = [{f'tag-{random.randint(0,10)}': result} for result in mocked_results_from_doc_eval]

    doc_evaluate_mock.side_effect = list(zip(mocked_results_from_doc_eval, mocked_results_by_tag_from_doc_eval))

    gold_spans = generate_non_overlapping_span_list('gold', 10)
    pred_spans = generate_non_overlapping_span_list('pred', 10)

    evaluator = CorpusEvaluator([gold_spans]*number_of_doc,[pred_spans]*number_of_doc)

    evaluator.evaluate()
    for tag, agg in evaluator.get_results_grouped_by_tags().items():
        agg.append_result_aggregator.has_calls([
            call(agg) for tag_res_for_doc in mocked_results_by_tag_from_doc_eval
                            for t, agg in tag_res_for_doc.items()  if tag==t 
        ])


@patch('seqnereval.evaluators.DocumentEvaluator.evaluate')
@patch('seqnereval.models.ResultAggregator.append_result_aggregator')
def test_corpus_evaluator_evaluate_get_results_by_doc(result_aggegator_mock, doc_evaluate_mock):
    number_of_doc = random.randint(2, 10)

    mocked_results_from_doc_eval = [generate_random_results_aggregator() for _ in range(number_of_doc)]
    mocked_results_by_tag_from_doc_eval = [{'tag': result} for result in mocked_results_from_doc_eval]

    doc_evaluate_mock.side_effect = list(zip(mocked_results_from_doc_eval, mocked_results_by_tag_from_doc_eval))

    gold_spans = generate_non_overlapping_span_list('gold', 10)
    pred_spans = generate_non_overlapping_span_list('pred', 10)

    evaluator = CorpusEvaluator([gold_spans]*number_of_doc,[pred_spans]*number_of_doc)

    evaluator.evaluate() 
    assert evaluator.get_results_by_doc() == mocked_results_from_doc_eval


def test_corpus_evaluator_get_results_before_eval():
    gold_spans = generate_non_overlapping_span_list('gold', 10)
    pred_spans = generate_non_overlapping_span_list('pred', 10)

    evaluator = CorpusEvaluator([gold_spans],[pred_spans])

    with pytest.raises(Exception) as result_requested_before_eval:
        evaluator.get_result()
    assert str(result_requested_before_eval.value)==("Evaluation has not been performed yet. "
    "Please call evaluate() before retrieving results.")

    
    with pytest.raises(Exception) as result_requested_before_eval:
        evaluator.get_results_grouped_by_tags()
    assert str(result_requested_before_eval.value)==("Evaluation has not been performed yet. "
    "Please call evaluate() before retrieving results.")

    
    with pytest.raises(Exception) as result_requested_before_eval:
        evaluator.get_results_by_doc()
    assert str(result_requested_before_eval.value)==("Evaluation has not been performed yet. "
    "Please call evaluate() before retrieving results.")
