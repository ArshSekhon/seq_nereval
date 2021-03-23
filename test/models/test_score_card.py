from seqnereval.models import ScoreCard

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

