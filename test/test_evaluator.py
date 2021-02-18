from seqnereval.models import NERGoldPredictedPair
from seqnereval import NERTagListEvaluator, NEREvaluator, NEREntitySpan
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
            {"tokens_spanned": ['John', 'Doe\'s'], "span_context": ['John', 'Doe\'s'],
                "entity_type": "PER", "start_idx": 1, "end_idx": 2},
            {"tokens_spanned": ['Basketball', 'Club'], "span_context": ['Basketball', 'Club'],
                "entity_type": "ORG", "start_idx": 3, "end_idx": 4},
        ],
        [
            {"tokens_spanned": ['Canada', 'Place'], "span_context": ['Canada', 'Place'],
                "entity_type": "LOC", "start_idx": 1, "end_idx": 2},
        ],
        [
            {'tokens_spanned': ['John'], 'span_context': ['John'], 'entity_type': 'PER', 'start_idx': 1,
             'end_idx': 1}
        ],
        [
            {'tokens_spanned': ['John', 'Doe'], 'span_context': ['John', 'Doe'],
                'entity_type': 'PER', 'start_idx': 0, 'end_idx': 1},
            {'tokens_spanned': ['Jenny', 'Doe'], 'span_context': ['Jenny', 'Doe'],
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
        NERTagListEvaluator([['X', 'X'], ['X', 'X']], [
                            ['O', 'O']], [['O', 'O']])

def test_ner_taglist_eval_tags_to_span_check_span_context():
    tokens = [
        ['The', 'John', 'Doe\'s', 'Basketball', 'Club'],
        ['The', 'Canada', 'Place', 'is', 'best', '.'],
    ]
    before = [
        ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"],
        ["O", "B-LOC", "I-LOC", "O", "O", "O"],
    ]

    expected = [
        [
            {"tokens_spanned": ['John', 'Doe\'s'], "span_context": ['The', 'John', 'Doe\'s', 'Basketball', 'Club'],
                "entity_type": "PER", "start_idx": 1, "end_idx": 2},
            {"tokens_spanned": ['Basketball', 'Club'], "span_context": ['John', 'Doe\'s', 'Basketball', 'Club'],
                "entity_type": "ORG", "start_idx": 3, "end_idx": 4},
        ],
        [
            {"tokens_spanned": ['Canada', 'Place'], "span_context": ['The', 'Canada', 'Place', 'is', 'best'],
                "entity_type": "LOC", "start_idx": 1, "end_idx": 2},
        ]
    ]
    evaluator = NERTagListEvaluator(tokens, before, before,2)
    gold_spans = [[span.__dict__ for span in span_list]
                  for span_list in evaluator.gold_entity_span_lists]
    pred_spans = [[span.__dict__ for span in span_list]
                  for span_list in evaluator.pred_entity_span_lists]

    # print(gold_spans)
    # print(pred_spans)
    assert gold_spans == expected
    assert pred_spans == expected

def test_ner_evaluate_missed_entity():
    missed_entity = NEREntitySpan('PER', 10, 32)
    gold_entities = [
        [
            missed_entity
        ]
    ]

    predicted_entities = [[]]
    evaluator = NEREvaluator(gold_entities, predicted_entities)
    res, _ = evaluator.evaluate()
    assert res.strict_match == {
        "correct": [],
        "incorrect": [],
        "partial": [],
        "missed": [missed_entity],
        "spurious": [],
        "possible": 1,
        "actual": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    assert res.type_match == {
        "correct": [],
        "incorrect": [],
        "partial": [],
        "missed": [missed_entity],
        "spurious": [],
        "possible": 1,
        "actual": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    assert res.partial_match == {
        "correct": [],
        "incorrect": [],
        "partial": [],
        "missed": [missed_entity],
        "spurious": [],
        "possible": 1,
        "actual": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    assert res.bounds_match == {
        "correct": [],
        "incorrect": [],
        "partial": [],
        "missed": [missed_entity],
        "spurious": [],
        "possible": 1,
        "actual": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

def test_ner_evaluate_unecessary_predicted_entity():
    unecessary_entity = NEREntitySpan('PER', 10, 32)

    gold_entities = [[]]
    predicted_entities = [[unecessary_entity]]

    evaluator = NEREvaluator(gold_entities, predicted_entities)
    res, _ = evaluator.evaluate()
    assert res.strict_match == {
        "correct": [],
        "incorrect": [],
        "partial": [],
        "missed": [],
        "spurious": [unecessary_entity],
        "possible": 0,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    assert res.type_match == {
        "correct": [],
        "incorrect": [],
        "partial": [],
        "missed": [],
        "spurious": [unecessary_entity],
        "possible": 0,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    assert res.partial_match == {
        "correct": [],
        "incorrect": [],
        "partial": [],
        "missed": [],
        "spurious": [unecessary_entity],
        "possible": 0,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    assert res.bounds_match == {
        "correct": [],
        "incorrect": [],
        "partial": [],
        "missed": [],
        "spurious": [unecessary_entity],
        "possible": 0,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

def test_ner_evaluator_type_match_span_match_overlap():
    predicted_entities = [
        [
            NEREntitySpan("PER", 24, 30),
            NEREntitySpan("LOC", 124, 134),
            NEREntitySpan("PER", 164, 174),
            NEREntitySpan("LOC", 197, 205),
            NEREntitySpan("LOC", 208, 219),
            NEREntitySpan("LOC", 225, 243),
        ]
    ]
    gold_entities = [
        [
            NEREntitySpan("PER", 59, 69),
            NEREntitySpan("LOC", 127, 134),
            NEREntitySpan("LOC", 164, 174),
            NEREntitySpan("LOC", 197, 205),
            NEREntitySpan("LOC", 208, 219),
            NEREntitySpan("MISC", 230, 240),
        ]
    ]

    evaluator = NEREvaluator(gold_entities, predicted_entities)
    res, _ = evaluator.evaluate()

    assert res.strict_match == {
        "correct": [NERGoldPredictedPair(NEREntitySpan("LOC", 197, 205), NEREntitySpan("LOC", 197, 205)),
                    NERGoldPredictedPair(NEREntitySpan("LOC", 208, 219), NEREntitySpan("LOC", 208, 219))],
        "incorrect": [NERGoldPredictedPair(NEREntitySpan("LOC", 127, 134), NEREntitySpan("LOC", 124, 134)),
                      NERGoldPredictedPair(NEREntitySpan("LOC", 164, 174), NEREntitySpan("PER", 164, 174)),
                      NERGoldPredictedPair(NEREntitySpan("MISC", 230, 240), NEREntitySpan("LOC", 225, 243))],
        "partial": [],
        "missed": [NEREntitySpan("PER", 59, 69)],
        "spurious": [NEREntitySpan("PER", 24, 30)],
        "possible": 6,
        "actual": 6,
        "precision": 0.3333333333333333,
        "recall": 0.3333333333333333,
        "f1": 0.3333333333333333,
    }

    assert res.type_match == {
        "correct": [NERGoldPredictedPair(NEREntitySpan("LOC", 127, 134), NEREntitySpan("LOC", 124, 134)),
                    NERGoldPredictedPair(NEREntitySpan("LOC", 197, 205), NEREntitySpan("LOC", 197, 205)),
                    NERGoldPredictedPair(NEREntitySpan("LOC", 208, 219), NEREntitySpan("LOC", 208, 219))],
        "incorrect": [NERGoldPredictedPair(NEREntitySpan("LOC", 164, 174), NEREntitySpan("PER", 164, 174)),
                      NERGoldPredictedPair(NEREntitySpan("MISC", 230, 240), NEREntitySpan("LOC", 225, 243))],
        "partial": [],
        "missed": [NEREntitySpan("PER", 59, 69)],
        "spurious": [NEREntitySpan("PER", 24, 30)],
        "possible": 6,
        "actual": 6,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }

    assert res.partial_match == {
        "correct": [NERGoldPredictedPair(NEREntitySpan("LOC", 164, 174), NEREntitySpan("PER", 164, 174)),
                    NERGoldPredictedPair(NEREntitySpan("LOC", 197, 205), NEREntitySpan("LOC", 197, 205)),
                    NERGoldPredictedPair(NEREntitySpan("LOC", 208, 219), NEREntitySpan("LOC", 208, 219))],
        "incorrect": [],
        "partial": [NERGoldPredictedPair(NEREntitySpan("LOC", 127, 134), NEREntitySpan("LOC", 124, 134)),
                    NERGoldPredictedPair(NEREntitySpan("MISC", 230, 240), NEREntitySpan("LOC", 225, 243))],
        "missed": [NEREntitySpan("PER", 59, 69)],
        "spurious": [NEREntitySpan("PER", 24, 30)],
        "possible": 6,
        "actual": 6,
        "precision": 0.66666666666666665,
        "recall": 0.6666666666666666,
        "f1": 0.6666666666666666,
    }

    assert res.bounds_match == {
        "correct": [NERGoldPredictedPair(NEREntitySpan("LOC", 164, 174), NEREntitySpan("PER", 164, 174)),
                    NERGoldPredictedPair(NEREntitySpan("LOC", 197, 205), NEREntitySpan("LOC", 197, 205)),
                    NERGoldPredictedPair(NEREntitySpan("LOC", 208, 219), NEREntitySpan("LOC", 208, 219))],
        "incorrect": [NERGoldPredictedPair(NEREntitySpan("LOC", 127, 134), NEREntitySpan("LOC", 124, 134)),
                      NERGoldPredictedPair(NEREntitySpan("MISC", 230, 240), NEREntitySpan("LOC", 225, 243))],
        "partial": [],
        "missed": [NEREntitySpan("PER", 59, 69)],
        "spurious": [NEREntitySpan("PER", 24, 30)],
        "possible": 6,
        "actual": 6,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }

def test_ner_evaluator_type_match_span_overlap():
    predicted_entities = [
        [
            NEREntitySpan("PER", 24, 30),
        ]
    ]
    gold_entities = [
        [
            NEREntitySpan("PER", 29, 69),
        ]
    ]

    evaluator = NEREvaluator(gold_entities, predicted_entities)
    res, _ = evaluator.evaluate()

    assert res.strict_match == {
        "correct": [],
        "incorrect": [NERGoldPredictedPair(NEREntitySpan("PER", 29, 69), NEREntitySpan("PER", 24, 30))],
        "partial": [],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    assert res.type_match == {
        "correct": [NERGoldPredictedPair(NEREntitySpan("PER", 29, 69), NEREntitySpan("PER", 24, 30))],
        "incorrect": [],
        "partial": [],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 1,
        "recall": 1,
        "f1": 1,
    }

    assert res.partial_match == {
        "correct": [],
        "incorrect": [],
        "partial": [NERGoldPredictedPair(NEREntitySpan("PER", 29, 69), NEREntitySpan("PER", 24, 30))],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }

    assert res.bounds_match == {
        "correct": [],
        "incorrect": [NERGoldPredictedPair(NEREntitySpan("PER", 29, 69), NEREntitySpan("PER", 24, 30))],
        "partial": [],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

def test_ner_evaluator_type_mismatch_span_exact():
    predicted_entities = [
        [
            NEREntitySpan("PER", 24, 30),
        ]
    ]
    gold_entities = [
        [
            NEREntitySpan("LOC", 24, 30),
        ]
    ]

    evaluator = NEREvaluator(gold_entities, predicted_entities)
    res, _ = evaluator.evaluate()

    assert res.strict_match == {
        "correct": [],
        "incorrect": [NERGoldPredictedPair(NEREntitySpan("LOC", 24, 30), NEREntitySpan("PER", 24, 30))],
        "partial": [],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    assert res.type_match == {
        "correct": [],
        "incorrect": [NERGoldPredictedPair(NEREntitySpan("LOC", 24, 30), NEREntitySpan("PER", 24, 30))],
        "partial": [],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    assert res.partial_match == {
        "correct": [NERGoldPredictedPair(NEREntitySpan("LOC", 24, 30), NEREntitySpan("PER", 24, 30))],
        "incorrect": [],
        "partial": [],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 1,
        "recall": 1,
        "f1": 1,
    }

    assert res.bounds_match == {
        "correct": [NERGoldPredictedPair(NEREntitySpan("LOC", 24, 30), NEREntitySpan("PER", 24, 30))],
        "incorrect": [],
        "partial": [],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 1,
        "recall": 1,
        "f1": 1,
    }

def test_ner_evaluator_type_mismatch_span_partial():
    predicted_entities = [
        [
            NEREntitySpan("PER", 24, 30),
        ]
    ]
    gold_entities = [
        [
            NEREntitySpan("LOC", 21, 26),
        ]
    ]

    evaluator = NEREvaluator(gold_entities, predicted_entities)
    res, _ = evaluator.evaluate()

    assert res.strict_match == {
        "correct": [],
        "incorrect": [NERGoldPredictedPair(NEREntitySpan("LOC", 21, 26), NEREntitySpan("PER", 24, 30))],
        "partial": [],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    assert res.type_match == {
        "correct": [],
        "incorrect": [NERGoldPredictedPair(NEREntitySpan("LOC", 21, 26), NEREntitySpan("PER", 24, 30))],
        "partial": [],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    assert res.partial_match == {
        "correct": [],
        "incorrect": [],
        "partial": [NERGoldPredictedPair(NEREntitySpan("LOC", 21, 26), NEREntitySpan("PER", 24, 30))],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }

    assert res.bounds_match == {
        "correct": [],
        "incorrect": [NERGoldPredictedPair(NEREntitySpan("LOC", 21, 26), NEREntitySpan("PER", 24, 30))],
        "partial": [],
        "missed": [],
        "spurious": [],
        "possible": 1,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

def test_ner_evaluator_summarize_results():
    predicted_entities = [
        [
            NEREntitySpan("PER", 24, 30),
        ]
    ]
    gold_entities = [
        [
            NEREntitySpan("LOC", 21, 26),
        ]
    ]

    evaluator = NEREvaluator(gold_entities, predicted_entities)
    res, _ = evaluator.evaluate()

    res.summarize_result() == {
    "strict_match": {
        "correct": 0,
        "incorrect": 1,
        "partial": 0,
        "missed": 0,
        "spurious": 0,
        "possible": 1,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    },

    "type_match": {
        "correct": 0,
        "incorrect": 1,
        "partial": 0,
        "missed": 0,
        "spurious": 0,
        "possible": 1,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    },

    "partial_match": {
        "correct": 0,
        "incorrect": 0,
        "partial": 1,
        "missed": 0,
        "spurious": 0,
        "possible": 1,
        "actual": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    },

    "bounds_match": {
        "correct": 0,
        "incorrect": 1,
        "partial": 0,
        "missed": 0,
        "spurious": 0,
        "possible": 1,
        "actual": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }}
