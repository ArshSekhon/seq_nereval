from unittest.mock import MagicMock
from pytest_mock import MockerFixture 
from copy import copy
import random

from seqnereval.models import ScoreCard, GoldPredictedPair, Span
from ..fixtures import (
    generate_random_gold_pred_span_pairs, 
    create_non_random_scorecard_fixture, 
    generate_random_scorecard_fixture, 
    create_non_random_type_or_partial_scorecard_fixture)


def test_score_card__init__():
    scorecard = ScoreCard()
    # check default values
    assert scorecard.correct == []
    assert scorecard.incorrect == []
    assert scorecard.partial == []
    assert scorecard.missed == []
    assert scorecard.spurious  == []

    assert scorecard.possible == 0
    assert scorecard.actual == 0
    assert scorecard.precision == 0
    assert scorecard.recall == 0
    assert scorecard.f1 == 0

    assert scorecard.is_partial_or_type_scorecard == False
    scorecard = ScoreCard(True)
    assert scorecard.is_partial_or_type_scorecard == True

def test_score_card_append_score_card(mocker: MockerFixture):
    scorecard = create_non_random_scorecard_fixture()

    scorecard_recalc_metrics = mocker.spy(scorecard, 'recalculate_metrics')

    # create copies needed for later assertions
    original_correct_fixture = copy(scorecard.correct)
    original_incorrect_fixture = copy(scorecard.incorrect)
    original_partial_fixture = copy(scorecard.partial)
    original_spurious_fixture = copy(scorecard.spurious)
    original_missed_fixture = copy(scorecard.missed)

    scorecard_to_append = generate_random_scorecard_fixture() 
    scorecard.appendScoreCard(scorecard_to_append)
    
    assert scorecard.correct == original_correct_fixture + scorecard_to_append.correct
    assert scorecard.incorrect == original_incorrect_fixture + scorecard_to_append.incorrect
    assert scorecard.partial == original_partial_fixture + scorecard_to_append.partial
    assert scorecard.spurious == original_spurious_fixture + scorecard_to_append.spurious
    assert scorecard.missed == original_missed_fixture + scorecard_to_append.missed

    scorecard_recalc_metrics.assert_called_once()

def test_score_card_get_summary():
    scorecard = create_non_random_scorecard_fixture()
    scorecard.recalculate_metrics()
    assert scorecard.get_summary() == {
        'actual': 11,
        'correct_counts': 3,
        'f1': 0.2608695652173913,
        'incorrect_counts': 3,
        'missed_counts': 3,
        'partial_counts': 3,
        'possible': 12,
        'precision': 0.2727272727272727,
        'recall': 0.25,
        'spurious_counts': 2
    }

def test_score_card_recalculate_metrice():
    scorecard = create_non_random_scorecard_fixture()
    scorecard.recalculate_metrics()

    assert scorecard.actual == 11
    assert scorecard.possible == 12
    assert scorecard.precision == 0.2727272727272727
    assert scorecard.recall == 0.25
    assert scorecard.f1 == 0.2608695652173913


    scorecard = create_non_random_type_or_partial_scorecard_fixture()

    assert scorecard.precision == 0.4090909090909091
    assert scorecard.recall == 0.375
    assert scorecard.f1 == 0.3913043478260869



    

