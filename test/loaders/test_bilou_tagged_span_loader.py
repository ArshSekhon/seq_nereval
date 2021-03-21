from seqnereval.loaders import BILOUTaggedSpanLoader

def test_bioes_tagged_span_loader_with_adjacent_single_and_separated_spans():
    tags = [
        'U-PER', 
        'O',
        'O',
        'B-LOC',
        'L-LOC', 
        'B-ORG',
        'I-ORG',
        'L-ORG',
        'O',
        'U-LOC',
        'O',
        'O',
        'B-PER',
        'I-PER',
        'I-PER',
        'L-PER'
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
    bilou_loader =  BILOUTaggedSpanLoader(tokens, tags)
    actual_spans =  bilou_loader.retreive_spans()
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
