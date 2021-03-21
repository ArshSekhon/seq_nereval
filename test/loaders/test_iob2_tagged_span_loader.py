import pytest
from seqnereval.loaders import IOB2TaggedSpanLoader

def test_iob2_tagged_span_loader_no_adjacent_spans():
    tags = [
        'B-PER',
        'O',
        'O',
        'O',
        'B-LOC',
        'I-LOC',
        'O',
        'B-LOC'
    ]

    tokens = [
        'Alex',
        'is',
        'going',
        'to',
        'Los',
        'Angeles',
        'in',
        'California'
    ]

    iob2_loader = IOB2TaggedSpanLoader(tokens, tags, 2)
    expected_span_dicts = [
        {
            'span_type': 'PER', 
            'start_idx': 0, 
            'end_idx': 0, 
            'spanned_tokens': ['Alex'], 
            'span_context': ['Alex', 'is', 'going']
        }, 
        {
            'span_type': 'LOC', 
            'start_idx': 4, 
            'end_idx': 5, 
            'spanned_tokens': ['Los', 'Angeles'], 
            'span_context': ['going', 'to', 'Los', 'Angeles', 'in', 'California']
        }, 
        {
            'span_type': 'LOC', 
            'start_idx': 7, 
            'end_idx': 7, 
            'spanned_tokens': ['California'], 
            'span_context': ['Angeles', 'in', 'California']
        }
    ]
    assert [span.__dict__ for span in iob2_loader.retreive_spans()] == expected_span_dicts


def test_iob_tagged_span_loader_with_adjacent_spans():
    tags = [
        'B-PER', 
        'O',
        'B-LOC',
        'I-LOC', 
        'B-ORG',
        'I-ORG',
        'I-ORG',
        'B-LOC',
    ]

    tokens = [
        'Alex', 
        'going', 
        'Los',
        'Angeles', 
        'California',
        'Club',
        'Ltd.',
        'California'
    ]

    iob2_loader = IOB2TaggedSpanLoader(tokens, tags, 2)
    expected_span_dicts = [
        {
            'span_type': 'PER', 
            'start_idx': 0, 
            'end_idx': 0, 
            'spanned_tokens': ['Alex'], 
            'span_context': ['Alex', 'going', 'Los']
        }, 
        {
            'span_type': 'LOC', 
            'start_idx': 2, 
            'end_idx': 3, 
            'spanned_tokens': ['Los', 'Angeles'], 
            'span_context': ['Alex', 'going', 'Los', 'Angeles', 'California', 'Club']
        }, 
        {
            'span_type': 'ORG', 
            'start_idx': 4, 
            'end_idx': 6, 
            'spanned_tokens': ['California', 'Club', 'Ltd.'], 
            'span_context': ['Los', 'Angeles', 'California', 'Club', 'Ltd.', 'California']
        },
        {
            'span_type': 'LOC', 
            'start_idx': 7, 
            'end_idx': 7, 
            'spanned_tokens': ['California'], 
            'span_context': ['Club', 'Ltd.', 'California']}
    ]

    assert [span.__dict__ for span in iob2_loader.retreive_spans()]==expected_span_dicts

def test_iob2_tagged_span_loader_unknown_tag_exception():
    tags = [
        'B-PER',
        'O',
        'O',
        'O',
        'B-LOC',
        'I-LOC',
        'O',
        'Z-LOC'
    ]

    tokens = [
        'Alex',
        'is',
        'going',
        'to',
        'Los',
        'Angeles',
        'in',
        'California'
    ]

    with pytest.raises(Exception) as unknown_tag_exception:
        iob2_loader = IOB2TaggedSpanLoader(tokens, tags)
        iob2_loader.retreive_spans()
    assert str(unknown_tag_exception.value)=='Unknown Token Tag: "Z-LOC" with prefix label: "Z"'

def test_iob2_tagged_span_loader_inside_before_begin_exception():
    tags = [
        'B-PER',
        'O',
        'O',
        'O',
        'I-LOC',
        'I-LOC',
        'O',
        'Z-LOC'
    ]

    tokens = [
        'Alex',
        'is',
        'going',
        'to',
        'Los',
        'Angeles',
        'in',
        'California'
    ]

    with pytest.raises(Exception) as inside_before_begin_exception:
        iob2_loader = IOB2TaggedSpanLoader(tokens, tags)
        iob2_loader.retreive_spans()
    assert str(inside_before_begin_exception.value)=='Encountered inside tag before a start tag at idx: "4".'