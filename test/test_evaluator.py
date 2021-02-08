from nereval import NERTagListEvaluator

def test_ner_taglist_eval_tags_to_span():
    before = [
        ["O", "B-LOC", "I-LOC", "B-LOC", "I-LOC", "O"],
        ["O", "B-GPE", "I-GPE", "B-GPE", "I-GPE", "O"],
    ]

    expected = [
        [
            {"entity_type": "LOC", "start_idx": 1, "end_idx": 2},
            {"entity_type": "LOC", "start_idx": 3, "end_idx": 4},
        ],
        [
            {"entity_type": "GPE", "start_idx": 1, "end_idx": 2},
            {"entity_type": "GPE", "start_idx": 3, "end_idx": 4},
        ],
    ]
    evaluator = NERTagListEvaluator([], before, before)
    gold_spans = [[span.__dict__ for span in span_list] for span_list in evaluator.gold_entity_span_lists]
    pred_spans = [[span.__dict__ for span in span_list] for span_list in evaluator.pred_entity_span_lists]
    

    assert gold_spans==expected 
    assert pred_spans==expected

if __name__=="__main__":
    test_ner_taglist_eval_tags_to_span()