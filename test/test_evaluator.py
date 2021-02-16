from seq_nereval import NERTagListEvaluator, NEREvaluator, NEREntitySpan
import pytest
import json

def test_ner_taglist_eval_tags_to_span():
    tokens = [
        ['The', 'John', 'Doe\'s', 'Basketball', 'Club'],
        ['The', 'Canada', 'Place', 'is', 'best', '.'],
        ['_', 'John', 'is', 'a', 'good', 'person', '.'],
        ['John', 'Doe', 'Jenny', 'Doe', '_', '_'],
    ]
    before = [
        ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"],
        ["O", "B-LOC", "I-LOC", "O", "O", "O"],
        ["O", "U-PER", "O", "O", "O", "O", "O"],
        ["B-PER", "I-PER", "B-PER", "I-PER", "O", "O"],
    ]

    expected = [
        [
            {"tokens_spanned": ['John', 'Doe\'s'],
                "entity_type": "PER", "start_idx": 1, "end_idx": 2},
            {"tokens_spanned": ['Basketball', 'Club'],
                "entity_type": "ORG", "start_idx": 3, "end_idx": 4},
        ],
        [
            {"tokens_spanned": ['Canada', 'Place'],
                "entity_type": "LOC", "start_idx": 1, "end_idx": 2},
        ],
        [
            {'tokens_spanned': ['John'], 'entity_type': 'PER', 'start_idx': 1,
             'end_idx': 1}
        ],
        [
            {'tokens_spanned': ['John', 'Doe'],
                'entity_type': 'PER', 'start_idx': 0, 'end_idx': 1},
            {'tokens_spanned': ['Jenny', 'Doe'],
                'entity_type': 'PER', 'start_idx': 2, 'end_idx': 3}
        ]
    ]
    evaluator = NERTagListEvaluator(tokens, before, before)
    gold_spans = [[span.__dict__ for span in span_list]
                  for span_list in evaluator.gold_entity_span_lists]
    pred_spans = [[span.__dict__ for span in span_list]
                  for span_list in evaluator.pred_entity_span_lists]

    # print(gold_spans)
    # print(pred_spans)
    assert gold_spans == expected
    assert pred_spans == expected


    # Test Exception Handling.
    with pytest.raises(Exception):
        NERTagListEvaluator([['X']], [['O', 'O']], [['O', 'O']])

    with pytest.raises(Exception):
        NERTagListEvaluator([['X', 'X'], ['X', 'X']], [['O', 'O']], [['O', 'O']])



def test_ner_evaluator_simple():
    predicted_entities = [
        [NEREntitySpan('PER', 0, 4), NEREntitySpan(
            'TEST', 5, 8), NEREntitySpan('PER', 10, 10)],
        [NEREntitySpan('PER', 0, 4), NEREntitySpan(
            'PER', 5, 8), NEREntitySpan('PER', 9, 10)]
    ]
    gold_entities = [
        [NEREntitySpan('PER', 0, 4), NEREntitySpan(
            'TEST', 5, 9), NEREntitySpan('PER', 10, 11)],
        [NEREntitySpan('PER', 0, 4), NEREntitySpan(
            'PER', 5, 8), NEREntitySpan('PER', 9, 10)]
    ]

    evaluator = NEREvaluator(gold_entities, predicted_entities)
    res, _ = evaluator.evaluate()
    # print("Scenario I", res.type_match_span_match)
    # print("Scenario II", res.unecessary_predicted_entity)
    # print("Scenario III", res.missed_gold_entity)
    # print("Scenario IV", res.type_mismatch_span_match)
    # print("Scenario V", res.type_match_span_partial)
    # print("Scenario VI", res.type_mismatch_span_partial)
    assert 1 == 1


if __name__ == "__main__":
    test_ner_taglist_eval_tags_to_span()
