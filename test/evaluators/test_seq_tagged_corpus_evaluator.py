from seqnereval.evaluators import SeqTaggedCorpusEvaluator
import pytest
from pytest_mock import MockerFixture
from unittest import mock

from ..fixtures import generate_random_span_list

def test_seq_tagged_corpus_evaluator__init__validation():
    tokens = ['Token']*20
    gold_spans = ['Gold_tag']*20
    pred_spans = ['Pred_tag']*20

    # validate tokens 2D list
    with pytest.raises(Exception) as validation_exception:
        SeqTaggedCorpusEvaluator(tokens, 
                                 [gold_spans], 
                                 [pred_spans], 
                                 SeqTaggedCorpusEvaluator.SupportedFormats.iob)
    assert str(validation_exception.value)==('SeqTaggedCorpusEvaluator creation failed.'
            ' tokens_grouped_by_docs is not a 2D List. Please pass a 2-D List into the'
            ' constructor containing token spans grouped by docs.')

    # validate gold_spans 2D list
    with pytest.raises(Exception) as validation_exception:
        SeqTaggedCorpusEvaluator([tokens], 
                                 gold_spans, 
                                 [pred_spans], 
                                 SeqTaggedCorpusEvaluator.SupportedFormats.iob)
    assert str(validation_exception.value)==('SeqTaggedCorpusEvaluator creation failed.'
            ' gold_tags_grouped_by_docs is not a 2D List. Please pass a 2-D List into the'
            ' constructor containing gold spans grouped by docs.')

    # validate pred_span 2D list
    with pytest.raises(Exception) as validation_exception:
        SeqTaggedCorpusEvaluator([tokens], 
                                 [gold_spans], 
                                 None, 
                                 SeqTaggedCorpusEvaluator.SupportedFormats.iob)
    assert str(validation_exception.value)==('SeqTaggedCorpusEvaluator creation failed.'
            ' predicted_tags_grouped_by_docs is not a 2D List. Please pass a 2-D List into the'
            ' constructor containing predicted spans grouped by docs.')
@mock.patch()
def test_seq_tagged_corpus_evaluator__init__(mocker:MockerFixture):
    pass