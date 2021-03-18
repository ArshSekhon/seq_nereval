from seqnereval.models import Span

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