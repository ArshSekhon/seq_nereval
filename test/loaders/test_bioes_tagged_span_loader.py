import pytest
from seqnereval.loaders import BIOESTaggedSpanLoader

def test_bioes_tagged_span_loader_with_adjacent_single_and_separated_spans():
    tags = [
        'S-PER', 
        'O',
        'O',
        'B-LOC',
        'E-LOC', 
        'B-ORG',
        'I-ORG',
        'E-ORG',
        'O',
        'S-LOC',
        'O',
        'O',
        'B-PER',
        'I-PER',
        'I-PER',
        'E-PER'
    ]

    tokens = [
        'Person1_Word1', 
        'Filler_Word1', 
        'Filler_Word2',
        'Loc1_Word1',
        'Loc1_Word2', 
        'Org1_Word1',
        'Org1_Word2',
        'Org1_Word3',
        'Filler_Word3',
        'Location2_Word1',
        'Filler_Word4',
        'Filler_Word5',
        'Person2_Word1', 
        'Person2_Word2',
        'Person2_Word3',
        'Person2_Word4',
    ]
    bioes_loader =  BIOESTaggedSpanLoader(tokens, tags)
    actual_spans =  bioes_loader.retreive_spans()
    expected_span_dicts = [
        {
            'span_type': 'PER', 
            'start_idx': 0, 
            'end_idx': 0, 
            'spanned_tokens': ['Person1_Word1'], 
            'span_context': ['Person1_Word1', 'Filler_Word1', 'Filler_Word2']
        }, 
        {
            'span_type': 'LOC', 
            'start_idx': 3, 
            'end_idx': 4, 
            'spanned_tokens': ['Loc1_Word1', 'Loc1_Word2'], 
            'span_context': ['Filler_Word1', 'Filler_Word2', 'Loc1_Word1', 'Loc1_Word2', 'Org1_Word1', 'Org1_Word2',]
        }, 
        {
            'span_type': 'ORG', 
            'start_idx': 5, 
            'end_idx': 7, 
            'spanned_tokens': ['Org1_Word1', 'Org1_Word2', 'Org1_Word3'], 
            'span_context': ['Loc1_Word1', 'Loc1_Word2', 'Org1_Word1', 'Org1_Word2', 'Org1_Word3', 'Filler_Word3', 'Location2_Word1']
        },
        {
            'span_type': 'LOC', 
            'start_idx': 9, 
            'end_idx': 9, 
            'spanned_tokens': ['Location2_Word1'], 
            'span_context': ['Org1_Word3', 'Filler_Word3', 'Location2_Word1', 'Filler_Word4', 'Filler_Word5']
        }, 
        {
            'span_type': 'PER', 
            'start_idx': 12, 
            'end_idx': 15, 
            'spanned_tokens': ['Person2_Word1', 'Person2_Word2', 'Person2_Word3', 'Person2_Word4'], 
            'span_context': ['Filler_Word4', 'Filler_Word5', 'Person2_Word1', 'Person2_Word2', 'Person2_Word3', 'Person2_Word4']
        }
    ]

    assert [span.__dict__ for span in actual_spans]==expected_span_dicts


def test_bioes_tagged_span_loader_outside_before_closing_exception():
    tags = [
        'S-PER', 
        'O',
        'O',
        'B-LOC',
        'O', 
        'B-ORG',
        'I-ORG',
        'E-ORG',
        'O',
        'S-LOC',
        'O',
        'O',
        'B-PER',
        'I-PER',
        'I-PER',
        'E-PER'
    ]

    tokens = [
        'Person1_Word1', 
        'Filler_Word1', 
        'Filler_Word2',
        'Loc1_Word1',
        'Filler_Word3', 
        'Org1_Word1',
        'Org1_Word2',
        'Org1_Word3',
        'Filler_Word3',
        'Location2_Word1',
        'Filler_Word4',
        'Filler_Word5',
        'Person2_Word1', 
        'Person2_Word2',
        'Person2_Word3',
        'Person2_Word4',
    ]
    bioes_loader =  BIOESTaggedSpanLoader(tokens, tags)

    with pytest.raises(Exception) as outside_before_closing_exception:
        bioes_loader.retreive_spans()
    assert str(outside_before_closing_exception.value)=='Encountered an outside tag before a span was closed at idx: 4.'


def test_bioes_tagged_span_loader_start_before_closing_exception():
    tags = [
        'S-PER', 
        'O',
        'O',
        'B-LOC',
        'I-LOC', 
        'B-ORG',
        'I-ORG',
        'E-ORG',
        'O',
        'S-LOC',
        'O',
        'O',
        'B-PER',
        'I-PER',
        'I-PER',
        'E-PER'
    ]

    tokens = [
        'Person1_Word1', 
        'Filler_Word1', 
        'Filler_Word2',
        'Loc1_Word1',
        'Loc1_Word2', 
        'Org1_Word1',
        'Org1_Word2',
        'Org1_Word3',
        'Filler_Word3',
        'Location2_Word1',
        'Filler_Word4',
        'Filler_Word5',
        'Person2_Word1', 
        'Person2_Word2',
        'Person2_Word3',
        'Person2_Word4',
    ]
    bioes_loader =  BIOESTaggedSpanLoader(tokens, tags)

    with pytest.raises(Exception) as start_before_closing_exception:
        bioes_loader.retreive_spans()
    assert str(start_before_closing_exception.value)=='Encountered a start tag before a span was closed at idx: 5.'


def test_bioes_tagged_span_loader_single_before_closing_exception():
    tags = [
        'S-PER', 
        'O',
        'O',
        'B-LOC',
        'S-LOC', 
        'B-ORG',
        'I-ORG',
        'E-ORG',
        'O',
        'S-LOC',
        'O',
        'O',
        'B-PER',
        'I-PER',
        'I-PER',
        'E-PER'
    ]

    tokens = [
        'Person1_Word1', 
        'Filler_Word1', 
        'Filler_Word2',
        'Loc1_Word1',
        'Loc1_Word2', 
        'Org1_Word1',
        'Org1_Word2',
        'Org1_Word3',
        'Filler_Word3',
        'Location2_Word1',
        'Filler_Word4',
        'Filler_Word5',
        'Person2_Word1', 
        'Person2_Word2',
        'Person2_Word3',
        'Person2_Word4',
    ]
    bioes_loader =  BIOESTaggedSpanLoader(tokens, tags)

    with pytest.raises(Exception) as single_before_closing_exception:
        bioes_loader.retreive_spans()
    assert str(single_before_closing_exception.value)=='Encountered a single start tag before a span was closed at idx: 4.'

def test_bioes_tagged_span_loader_inside_before_start_exception():
    tags = [
        'S-PER', 
        'O',
        'O',
        'I-LOC',
        'I-LOC', 
        'B-ORG',
        'I-ORG',
        'E-ORG',
        'O',
        'S-LOC',
        'O',
        'O',
        'B-PER',
        'I-PER',
        'I-PER',
        'E-PER'
    ]

    tokens = [
        'Person1_Word1', 
        'Filler_Word1', 
        'Filler_Word2',
        'Loc1_Word1',
        'Loc1_Word2', 
        'Org1_Word1',
        'Org1_Word2',
        'Org1_Word3',
        'Filler_Word3',
        'Location2_Word1',
        'Filler_Word4',
        'Filler_Word5',
        'Person2_Word1', 
        'Person2_Word2',
        'Person2_Word3',
        'Person2_Word4',
    ]
    bioes_loader =  BIOESTaggedSpanLoader(tokens, tags)

    with pytest.raises(Exception) as single_before_closing_exception:
        bioes_loader.retreive_spans()
    assert str(single_before_closing_exception.value)=='Encountered an inside tag before a start tag at idx: 3.'

def test_bioes_tagged_span_loader_end_before_start_exception():
    tags = [
        'S-PER', 
        'O',
        'O',
        'E-LOC',
        'O', 
        'B-ORG',
        'I-ORG',
        'E-ORG',
        'O',
        'S-LOC',
        'O',
        'O',
        'B-PER',
        'I-PER',
        'I-PER',
        'E-PER'
    ]

    tokens = [
        'Person1_Word1', 
        'Filler_Word1', 
        'Filler_Word2',
        'Loc1_Word1',
        'Filler_Word4', 
        'Org1_Word1',
        'Org1_Word2',
        'Org1_Word3',
        'Filler_Word3',
        'Location2_Word1',
        'Filler_Word4',
        'Filler_Word5',
        'Person2_Word1', 
        'Person2_Word2',
        'Person2_Word3',
        'Person2_Word4',
    ]
    bioes_loader =  BIOESTaggedSpanLoader(tokens, tags)

    with pytest.raises(Exception) as single_before_closing_exception:
        bioes_loader.retreive_spans()
    assert str(single_before_closing_exception.value)=='Encountered an end tag before a start tag at idx: 3.'


def test_bioes_tagged_span_loader_unknown_prefix_exception():
    tags = [
        'S-PER', 
        'O',
        'O',
        'Y-LOC',
        'O', 
        'B-ORG',
        'I-ORG',
        'E-ORG',
        'O',
        'S-LOC',
        'O',
        'O',
        'B-PER',
        'I-PER',
        'I-PER',
        'E-PER'
    ]

    tokens = [
        'Person1_Word1', 
        'Filler_Word1', 
        'Filler_Word2',
        'Loc1_Word1',
        'Filler_Word4', 
        'Org1_Word1',
        'Org1_Word2',
        'Org1_Word3',
        'Filler_Word3',
        'Location2_Word1',
        'Filler_Word4',
        'Filler_Word5',
        'Person2_Word1', 
        'Person2_Word2',
        'Person2_Word3',
        'Person2_Word4',
    ]
    bioes_loader =  BIOESTaggedSpanLoader(tokens, tags)

    with pytest.raises(Exception) as single_before_closing_exception:
        bioes_loader.retreive_spans()
    assert str(single_before_closing_exception.value)=='Unknown Token Tag: "Y-LOC" with an unknown prefix label: "Y" at idx: 3.'


def test_bioes_tagged_span_loader_tags_end_before_span_close_exception():
    tags = [
        'S-PER', 
        'O',
        'O',
        'B-LOC',
        'E-LOC', 
        'B-ORG',
        'I-ORG',
        'E-ORG',
        'O',
        'S-LOC',
        'O',
        'O',
        'B-PER',
        'I-PER',
        'I-PER',
    ]

    tokens = [
        'Person1_Word1', 
        'Filler_Word1', 
        'Filler_Word2',
        'Loc1_Word1',
        'Loc1_Word2', 
        'Org1_Word1',
        'Org1_Word2',
        'Org1_Word3',
        'Filler_Word3',
        'Location2_Word1',
        'Filler_Word4',
        'Filler_Word5',
        'Person2_Word1', 
        'Person2_Word2',
        'Person2_Word3',
    ]
    bioes_loader =  BIOESTaggedSpanLoader(tokens, tags)

    with pytest.raises(Exception) as single_before_closing_exception:
        bioes_loader.retreive_spans()
    assert str(single_before_closing_exception.value)=='Tag List ended before a span was closed at the idx: 14'
