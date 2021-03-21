import pytest
from seqnereval.loaders import IOBTaggedSpanLoader

def test_iob_tagged_span_mismatched_tokens_tags_exceptions():
    tags = [
        'I-PER',
        'O',
        'O',
        'O',
        'I-LOC',
        'I-LOC',
        'O',
        'I-LOC'
    ]

    tokens = [
        'Alex',
        'is',
        'going',
        'to',
        'Los',
    ]
    with pytest.raises(Exception) as mismatch_exception:
        IOBTaggedSpanLoader(tags, tokens)
    assert str(mismatch_exception.value)=='Number of tokens and tags is not the same.'

def test_iob_tagged_span_loader_unknown_tag_exception():
    tags = [
        'I-PER',
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

    with pytest.raises(Exception) as unknown_tag_exception:
        iob_loader = IOBTaggedSpanLoader(tokens, tags)
        iob_loader.retreive_spans()
    assert str(unknown_tag_exception.value)=='Unknown Token Tag: "Z-LOC" with prefix label: "Z" at idx: 7.'


def test_iob_tagged_span_loader_get_prefix_tag_too_short_exception():
    tags = [
        'I-PER',
        'Z-',
        'O',
        'O' 
    ]

    tokens = [
        'Alex',
        'is',
        'going',
        'to', 
    ]

    with pytest.raises(Exception) as tag_to_short_exception:
        iob_loader = IOBTaggedSpanLoader(tokens, tags)
        iob_loader.get_prefix_for_tag_at(1)
    assert str(tag_to_short_exception.value)=='Tag length too short cannot extract a prefix from the tag "Z-" at idx: 1.'


def test_iob_tagged_span_loader_get_label_exception():
    tags = [
        'I-PER',
        'Z-',
        'O',
        'O' 
    ]

    tokens = [
        'Alex',
        'is',
        'going',
        'to', 
    ]

    with pytest.raises(Exception) as tag_to_short_exception:
        iob_loader = IOBTaggedSpanLoader(tokens, tags)
        iob_loader.get_label_for_tag_at(1)
    assert str(tag_to_short_exception.value)=='Tag length too short cannot extract a label from the tag "Z-" at idx: 1.'

    
    with pytest.raises(Exception) as outside_tag_exception:
        iob_loader = IOBTaggedSpanLoader(tokens, tags)
        iob_loader.get_label_for_tag_at(3)
    assert str(outside_tag_exception.value)=='Cannot retrieve label for a token tagged with "O" at idx: 3.'

def test_iob_tagged_span_loader_different_labels_in_span_exception():
    tags = [
        'I-PER', 
        'O',
        'I-LOC',
        'I-LOC', 
        'B-ORG',
        'I-LOC',
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

    iob_loader = IOBTaggedSpanLoader(tokens, tags, 2) 
    with pytest.raises(Exception) as different_labels_in_span_exception:
        iob_loader.retreive_spans()
    assert (str(different_labels_in_span_exception.value) == 
        "Tokens with different labels found within the same span for idx range (4, 7).")


def test_iob_tagged_span_loader_no_adjacent_spans():
    tags = [
        'I-PER',
        'O',
        'O',
        'O',
        'I-LOC',
        'I-LOC',
        'O',
        'I-LOC'
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

    iob_loader = IOBTaggedSpanLoader(tokens, tags, 2)
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
    assert [span.__dict__ for span in iob_loader.retreive_spans()] == expected_span_dicts

def test_iob_tagged_span_loader_with_adjacent_spans():
    tags = [
        'I-PER', 
        'O',
        'I-LOC',
        'I-LOC', 
        'B-LOC'
    ]

    tokens = [
        'Alex', 
        'going', 
        'Los',
        'Angeles', 
        'California'
    ]

    iob_loader = IOBTaggedSpanLoader(tokens, tags, 2)
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
            'span_context': ['Alex', 'going', 'Los', 'Angeles', 'California']
        }, 
        {
            'span_type': 'LOC', 
            'start_idx': 4, 
            'end_idx': 4, 
            'spanned_tokens': ['California'], 
            'span_context': ['Los', 'Angeles','California']
        }
    ]
    assert [span.__dict__ for span in iob_loader.retreive_spans()] == expected_span_dicts



def test_iob_tagged_span_loader_with_adjacent_spans_longer():
    tags = [
        'I-PER', 
        'O',
        'I-LOC',
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

    iob_loader = IOBTaggedSpanLoader(tokens, tags, 2)
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

    assert [span.__dict__ for span in iob_loader.retreive_spans()]==expected_span_dicts