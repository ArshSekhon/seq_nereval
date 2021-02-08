from seq_nereval import NERTagListEvaluator, NEREvaluator, NEREntitySpan
import json

def test_ner_evaluator_simple():
    predicted_entities = [
        [NEREntitySpan('PER',0,4),NEREntitySpan('TEST',5,8),NEREntitySpan('PER',10,10)],
        [NEREntitySpan('PER',0,4),NEREntitySpan('PER',5,8),NEREntitySpan('PER',9,10)]
    ]
    gold_entities = [
        [NEREntitySpan('PER',0,4),NEREntitySpan('TEST',5,9),NEREntitySpan('PER',10,11)],
        [NEREntitySpan('PER',0,4),NEREntitySpan('PER',5,8),NEREntitySpan('PER',9,10)]
    ]

    evaluator = NEREvaluator(gold_entities, predicted_entities)
    res, _ = evaluator.calculate_metrics_for_doc(evaluator.gold_entity_span_lists[0], evaluator.pred_entity_span_lists[0])
    print("Scenario I", res.type_match_span_match)
    print("Scenario II", res.unecessary_predicted_entity)
    print("Scenario III", res.missed_gold_entity)
    print("Scenario IV", res.type_mismatch_span_match)
    print("Scenario V", res.type_match_span_partial)
    print("Scenario VI", res.type_mismatch_span_partial)


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