from __future__ import annotations

from . import Span

class GoldPredictedPair:
    """Pair of gold and predicted spans
    """

    def __init__(self, gold_span: Span, predicted_span: Span) -> None:
        """Construct a new GoldPredicted Pair.

        Args:
            gold_span (Span): Golden span.
            predicted_span (Span): Predicted span.
        """
        self.gold_span = gold_span
        self.predicted_span = predicted_span

    def __str__(self) -> str:
        return f'{{Gold: {self.gold_span}, Predicted: {self.predicted_span}}}'

    def __repr__(self) -> str:
        return f'{{Gold: {self.gold_span}, Predicted: {self.predicted_span}}}'

    def __eq__(self, o: GoldPredictedPair) -> bool:
        return self.gold_span == o.gold_span and self.predicted_span == o.predicted_span


