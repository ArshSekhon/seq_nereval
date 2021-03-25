import pytest
from seqnereval.models import Span
from ..fixtures import generate_random_span_list

def test_span__init__():
    # Scenario 1
    span = Span('tag', 100, 200)

    # assert span types
    assert span.span_type == 'tag'
    assert span.start_idx == 100
    assert span.end_idx == 200
    # assert defaults
    assert span.spanned_tokens == None
    assert span.span_context == None
    
    # Scenario 2
    tokens = generate_random_span_list('token', 10)
    span = Span('tag', 10, 20, tokens)

    # assert assigned value
    assert span.spanned_tokens == tokens
    # assert default value 
    assert span.span_context == span.spanned_tokens

    # Scenario 3
    tokens = generate_random_span_list('token', 10)
    context_tokens = generate_random_span_list('token', 20)
    span = Span('tag', 10, 20, tokens, context_tokens)

    # assert assigned value
    assert span.spanned_tokens == tokens
    # assert assigned value 
    assert span.span_context == context_tokens



def test_Span_bounds_same_tokens_as():
    span = Span('test', 10, 15)

    same_bounds_span = Span('test', 10, 15)
    assert span.bounds_same_tokens_as(same_bounds_span) == True

    other_span_contained_inside = Span('test', 11, 12)
    assert span.bounds_same_tokens_as(other_span_contained_inside) == False

    other_span_overlapping_left = Span('test', 7, 10)
    assert span.bounds_same_tokens_as(other_span_overlapping_left) == False

    other_span_overlapping_right = Span('test', 15, 20)
    assert span.bounds_same_tokens_as(other_span_overlapping_right) == False

    other_span_non_overlapping_left = Span('test', 0, 7)
    assert span.bounds_same_tokens_as(other_span_non_overlapping_left) == False

    entity_non_overlap_right = Span('test', 20, 27)
    assert span.bounds_same_tokens_as(entity_non_overlap_right) == False

def test_Span_overlaps_with():
    span = Span('test', 10, 15)

    same_bounds_span = Span('test', 10, 15)
    assert span.overlaps_with(same_bounds_span) == True

    other_span_contained_inside = Span('test', 11, 12)
    assert span.overlaps_with(other_span_contained_inside) == True

    other_span_overlapping_left = Span('test', 7, 10)
    assert span.overlaps_with(other_span_overlapping_left) == True

    other_span_overlapping_right = Span('test', 15, 20)
    assert span.overlaps_with(other_span_overlapping_right) == True

    other_span_non_overlapping_left = Span('test', 0, 7)
    assert span.overlaps_with(other_span_non_overlapping_left) == False

    entity_non_overlap_right = Span('test', 20, 27)
    assert span.overlaps_with(entity_non_overlap_right) == False

def test_Span__str__():
    assert Span('test',1,2,['X1','X2']).__str__() == "(Type: \"test\", Token Span IDX:(1, 2), Tokens:['X1', 'X2'], Context:['X1', 'X2'])"

def test_Span__repr__():
    assert Span('test',1,2,['X1','X2']).__repr__() == "(Type: \"test\", Token Span IDX:(1, 2), Tokens:['X1', 'X2'], Context:['X1', 'X2'])"

def test_Span__hash__():
    assert Span('test',1,2,['X1','X2']) in set([Span('test',1,2,['X1','X2'])])
    assert Span('test',1,3,['X1','X2']) not in set([Span('test',1,2,['X1','X2'])])

def test_span_ends_after_end_of():
    span = Span('test_tag', 10, 20)
    span_before = Span('test_tag', 6, 8)
    span_before_overlap = Span('test_tag', 8, 10)
    span_contained_in = Span('test_tag', 12, 15)
    span_after_overlap = Span('test_tag', 20, 22)
    span_after  = Span('test_tag', 22, 25)

    assert span.ends_after_end_of(span_before) == True
    assert span.ends_after_end_of(span_before_overlap) == True
    assert span.ends_after_end_of(span_contained_in) == True
    assert span.ends_after_end_of(span_after_overlap) == False
    assert span.ends_after_end_of(span_after) == False

def test_span_ends_before_start_of():
    span = Span('test_tag', 10, 20)
    span_before = Span('test_tag', 6, 8)
    span_before_overlap = Span('test_tag', 8, 10)
    span_contained_in = Span('test_tag', 12, 15)
    span_after_overlap = Span('test_tag', 20, 22)
    span_after  = Span('test_tag', 22, 25)

    assert span.ends_before_start_of(span_before) == False
    assert span.ends_before_start_of(span_before_overlap) == False
    assert span.ends_before_start_of(span_contained_in) == False
    assert span.ends_before_start_of(span_after_overlap) == False
    assert span.ends_before_start_of(span_after) == True

def test_span_starts_before_start_of():
    span = Span('test_tag', 10, 20)
    span_before = Span('test_tag', 6, 8)
    span_before_overlap = Span('test_tag', 8, 10)
    span_contained_in = Span('test_tag', 12, 15)
    span_after_overlap = Span('test_tag', 20, 22)
    span_after  = Span('test_tag', 22, 25)

    assert span.starts_before_start_of(span_before) == False
    assert span.starts_before_start_of(span_before_overlap) == False
    assert span.starts_before_start_of(span_contained_in) == True
    assert span.starts_before_start_of(span_after_overlap) == True
    assert span.starts_before_start_of(span_after) == True


def test_span_starts_after_end_of():
    span = Span('test_tag', 10, 20)
    span_before = Span('test_tag', 6, 8)
    span_before_overlap = Span('test_tag', 8, 10)
    span_contained_in = Span('test_tag', 12, 15)
    span_after_overlap = Span('test_tag', 20, 22)
    span_after  = Span('test_tag', 22, 25)

    assert span.starts_after_end_of(span_before) == True
    assert span.starts_after_end_of(span_before_overlap) == False
    assert span.starts_after_end_of(span_contained_in) == False
    assert span.starts_after_end_of(span_after_overlap) == False
    assert span.starts_after_end_of(span_after) == False

def test_span_start_end_idx_validation():
    with pytest.raises(Exception) as exception_raised:
        Span('tag',10,9)
    
    assert str(exception_raised.value)=='Start IDX for a span cannot be > End IDX.'


