from seqnereval.models import GoldPredictedPair, Span

def test_GoldPredictedPair__init__():
    dummyGoldSpan = Span("test",0,10)
    dummyPredictedSpan =  Span("test", 0, 11)

    pair = GoldPredictedPair(dummyGoldSpan, dummyPredictedSpan)

    assert dummyGoldSpan == pair.gold_span
    assert dummyPredictedSpan == pair.predicted_span

def test_GoldPredictedPair__str__():
    dummyGoldSpan = Span("test",0,10)
    dummyPredictedSpan =  Span("test", 0, 11)

    pair = GoldPredictedPair(dummyGoldSpan, dummyPredictedSpan)
    assert pair.__str__()==f'{{Gold: {dummyGoldSpan}, Predicted: {dummyPredictedSpan}}}'
    

def test_GoldPredictedPair__repr__():
    dummyGoldSpan = Span("test",0,10)
    dummyPredictedSpan =  Span("test", 0, 11)

    pair = GoldPredictedPair(dummyGoldSpan, dummyPredictedSpan)
    assert pair.__repr__()==f'{{Gold: {dummyGoldSpan}, Predicted: {dummyPredictedSpan}}}'


def test_GoldPredictedPair__eq__():
    dummyGoldSpan1 = Span("test",0,10)
    dummyPredictedSpan1 =  Span("test", 0, 11)

    dummyGoldSpan2 = Span("test",0,10)
    dummyPredictedSpan2 =  Span("test", 0, 11)

    pair1 = GoldPredictedPair(Span("test",0,10), Span("test", 0, 11))
    pair2 = GoldPredictedPair(Span("test",0,10), Span("test", 0, 11))

    pair3 = GoldPredictedPair(Span("test", 0, 12), Span("test", 0, 11))
    pair4 = GoldPredictedPair(Span("test", 0, 10), Span("test", 0, 12))
    assert pair1 == pair2
    assert pair1 != pair3
    assert pair1 != pair4
