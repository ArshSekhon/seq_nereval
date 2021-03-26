from seqnereval.evaluators import TaggedSeqEvaluator
from seqnereval.models import Span
from seqnereval.loaders import IOBTaggedSpanLoader
import pytest
from pytest_mock import MockerFixture
from unittest.mock import patch, call
import pprint

from ..fixtures import generate_non_overlapping_span_list


@patch.object(IOBTaggedSpanLoader, '__init__', return_value=None)
@patch.object(IOBTaggedSpanLoader, 'retreive_spans', side_effect=[[Span('gold',1,1)],[Span('predicted',1,1)]])
def test_seq_tagged_corpus_evaluator__init__(mock_iob_span_loader, retreive_spans_mock):
    tokens = ['Token']*20
    gold_spans = ['Gold_tag']*20
    pred_spans = ['Pred_tag']*20    

    evaluator = TaggedSeqEvaluator(tokens, 
                                    gold_spans, 
                                    pred_spans, 
                                    TaggedSeqEvaluator.SupportedFormats.iob)
    
    assert mock_iob_span_loader.has_calls([call(tokens, gold_spans), call(tokens, pred_spans)]) 
    assert retreive_spans_mock.call_count == 2

    assert evaluator.gold_spans == [Span('gold',1,1)]
    assert evaluator.predicted_spans == [Span('predicted',1,1)]

# Test Loader Selectors

@patch('seqnereval.loaders.IOBTaggedSpanLoader')
def test_seq_tagged_corpus_evaluator_test_loader_selection_iob(mock_iob_span_loader):
    tokens = ['Token']*20
    gold_spans = ['O']*20
    pred_spans = ['O']*20    

    TaggedSeqEvaluator(tokens, 
                        gold_spans, 
                        pred_spans, 
                        TaggedSeqEvaluator.SupportedFormats.iob)
    
    assert mock_iob_span_loader.has_calls([call(tokens, gold_spans), call(tokens, pred_spans)]) 

@patch('seqnereval.loaders.IOB2TaggedSpanLoader')
def test_seq_tagged_corpus_evaluator_test_loader_selection_iob2(mock_iob2_span_loader):
    tokens = ['Token']*20
    gold_spans = ['O']*20
    pred_spans = ['O']*20    

    TaggedSeqEvaluator(tokens, 
                        gold_spans, 
                        pred_spans, 
                        TaggedSeqEvaluator.SupportedFormats.iob2)
    
    assert mock_iob2_span_loader.has_calls([call(tokens, gold_spans), call(tokens, pred_spans)]) 
    
@patch('seqnereval.loaders.BIOESTaggedSpanLoader')
def test_seq_tagged_corpus_evaluator_test_loader_selection_bioes(mock_bioes_span_loader):
    tokens = ['Token']*20
    gold_spans = ['O']*20
    pred_spans = ['O']*20    

    TaggedSeqEvaluator(tokens, 
                        gold_spans, 
                        pred_spans, 
                        TaggedSeqEvaluator.SupportedFormats.bioes)
    
    assert mock_bioes_span_loader.has_calls([call(tokens, gold_spans), call(tokens, pred_spans)]) 
    
@patch('seqnereval.loaders.BILOUTaggedSpanLoader')
def test_seq_tagged_corpus_evaluator_test_loader_selection_bilou(mock_bilou_span_loader):
    tokens = ['Token']*20
    gold_spans = ['O']*20
    pred_spans = ['O']*20    

    TaggedSeqEvaluator(tokens, 
                        gold_spans, 
                        pred_spans, 
                        TaggedSeqEvaluator.SupportedFormats.bioes)
    
    assert mock_bilou_span_loader.has_calls([call(tokens, gold_spans), call(tokens, pred_spans)]) 
