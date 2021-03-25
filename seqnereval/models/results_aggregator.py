from __future__ import annotations
from typing import List, Tuple, Dict
from . import Span, GoldPredictedPair, ScoreCard

class ResultAggregator:
    def __init__(self):
        """
        Constructor for ResultAggregator 
        """
        self.strict_match = ScoreCard()
        self.type_match = ScoreCard(is_partial_or_type_scorecard=True)
        self.partial_match = ScoreCard(is_partial_or_type_scorecard=True)
        self.bounds_match = ScoreCard()

        self.type_match_bounds_match: List[GoldPredictedPair] = []
        self.unecessary_predicted_span: List[Span] = []
        self.missed_gold_span: List[Span] = []
        self.type_mismatch_bounds_match: List[GoldPredictedPair] = []
        self.type_match_bounds_partial: List[GoldPredictedPair] = []
        self.type_mismatch_bounds_partial: List[GoldPredictedPair] = []

    def summarize_result(self):
        """Summarizes the results into numbers.
        """

        return {
            "strict_match": self.strict_match.get_summary(),
            "type_match": self.type_match.get_summary(),
            "partial_match": self.partial_match.get_summary(),
            "bounds_match": self.bounds_match.get_summary(),
            "type_match_bounds_match": len(self.type_match_bounds_match),
            "unecessary_predicted_span": len(self.unecessary_predicted_span),
            "missed_gold_span": len(self.missed_gold_span),
            "type_mismatch_bounds_match": len(self.type_mismatch_bounds_match),
            "type_match_bounds_partial": len(self.type_match_bounds_partial),
            "type_mismatch_bounds_partial": len(self.type_mismatch_bounds_partial)
        }

    def append_result_aggregator(self, otherResultAggregator: ResultAggregator) -> None:
        """Appends the results obtained from a different evaluation.

        Args:
            otherResultAggregator (ResultAggregator): Result to be appended.
        """

        self.strict_match.appendScoreCard(otherResultAggregator.strict_match)
        self.type_match.appendScoreCard(otherResultAggregator.type_match)
        self.partial_match.appendScoreCard(otherResultAggregator.partial_match)
        self.bounds_match.appendScoreCard(otherResultAggregator.bounds_match)

        self.type_match_bounds_match.extend(otherResultAggregator.type_match_bounds_match)
        self.unecessary_predicted_span.extend(otherResultAggregator.unecessary_predicted_span)
        self.missed_gold_span.extend(otherResultAggregator.missed_gold_span)
        self.type_mismatch_bounds_match.extend(otherResultAggregator.type_mismatch_bounds_match)
        self.type_match_bounds_partial.extend(otherResultAggregator.type_match_bounds_partial)
        self.type_mismatch_bounds_partial.extend(otherResultAggregator.type_mismatch_bounds_partial)

        self.recalculate_metrics_for_all_scorecards() 

    # Scenario I
    def add_type_match_bounds_match(self, gold_span: Span, pred_span: Span) -> None:
        """Add Gold and Predicted span pair to Scenario I aggregator: both type and bounds match.

        Args:
            gold_span (Span): Golden Span.
            pred_span (Span): Predicted Span.
        """

        self.type_match_bounds_match.append(GoldPredictedPair(gold_span, pred_span))

        self.strict_match.correct.append(GoldPredictedPair(gold_span, pred_span))
        self.type_match.correct.append(GoldPredictedPair(gold_span, pred_span))
        self.partial_match.correct.append(GoldPredictedPair(gold_span, pred_span))
        self.bounds_match.correct.append(GoldPredictedPair(gold_span, pred_span))

        self.recalculate_metrics_for_all_scorecards()

    # Scenario II

    def add_unecessary_predicted_span(self, uncessary_pred_span: Span) -> None:
        """Add wrongly predicted span to the Scenario II aggregate: predicted 
           span doesn't exist in golden dataset.

        Args:
            uncessary_pred_span (Span): Span that was wrongly predicted.
        """

        self.unecessary_predicted_span.append(uncessary_pred_span)

        self.strict_match.spurious.append(uncessary_pred_span)
        self.type_match.spurious.append(uncessary_pred_span)
        self.partial_match.spurious.append(uncessary_pred_span)
        self.bounds_match.spurious.append(uncessary_pred_span)

        self.recalculate_metrics_for_all_scorecards()

    # Scenario III
    def add_missed_gold_span(self, missed_gold_span: Span) -> None:
        """Add missed span to Scenario III aggregate: missed span that 
            wasn't predicted

        Args:
            missed_gold_span (Span): Span that wasn't predicted.
        """

        self.missed_gold_span.append(missed_gold_span)

        self.strict_match.missed.append(missed_gold_span)
        self.type_match.missed.append(missed_gold_span)
        self.partial_match.missed.append(missed_gold_span)
        self.bounds_match.missed.append(missed_gold_span)

        self.recalculate_metrics_for_all_scorecards()

    # Scenario IV
    def add_type_mismatch_bounds_match(self, gold_span: Span, pred_span: Span) -> None:
        """Add Gold and predicted pair for which bounds were predicted correctly but the 
          type was incorrect.

        Args:
            gold_span (Span): Gold span for which which bounds were correct
                                            but the type was not.
            pred_span (Span): Predicted span for which bounds were correct
                                            but the type was not.
        """

        self.type_mismatch_bounds_match.append(GoldPredictedPair(gold_span, pred_span))

        self.strict_match.incorrect.append(GoldPredictedPair(gold_span, pred_span))
        self.type_match.incorrect.append(GoldPredictedPair(gold_span, pred_span))
        self.partial_match.correct.append(GoldPredictedPair(gold_span, pred_span))
        self.bounds_match.correct.append(GoldPredictedPair(gold_span, pred_span))

        self.recalculate_metrics_for_all_scorecards()

    # Scenario V
    def add_type_match_bounds_partial(self, gold_span: Span, pred_span: Span) -> None:
        """Add Gold and predicted span pair for which bounds were predicted partially correctly whereas the 
             type was predicted correctly. 

        Args:
            gold_span (Span): Gold span for which bounds were predicted partially correctly
                whereas the type was predicted correctly. 
            pred_span (Span): Predicted span for which bounds were predicted partially correctly
                whereas the type was predicted correctly. 
        """

        self.type_match_bounds_partial.append(GoldPredictedPair(gold_span, pred_span))

        self.strict_match.incorrect.append(GoldPredictedPair(gold_span, pred_span))
        self.type_match.correct.append(GoldPredictedPair(gold_span, pred_span))
        self.partial_match.partial.append(GoldPredictedPair(gold_span, pred_span))
        self.bounds_match.incorrect.append(GoldPredictedPair(gold_span, pred_span))

        self.recalculate_metrics_for_all_scorecards()

    # Scenario VI
    def add_type_mismatch_bounds_partial(self, gold_span: Span, pred_span: Span) -> None:
        """Add Gold and predicted span pair for which bounds were predicted partially correctly and the
                type was incorrect.

        Args:
            gold_span (Span): Gold span for which bounds were predicted partially correctly
                whereas the type was predicted incorrectly. 
            pred_span (Span): Predicted Entity span for which bounds were predicted partially correctly
                whereas the type was predicted incorrectly. 
        """

        self.type_mismatch_bounds_partial.append(
            GoldPredictedPair(gold_span, pred_span))

        self.strict_match.incorrect.append(GoldPredictedPair(gold_span, pred_span))
        self.type_match.incorrect.append(GoldPredictedPair(gold_span, pred_span))
        self.partial_match.partial.append(GoldPredictedPair(gold_span, pred_span))
        self.bounds_match.incorrect.append(GoldPredictedPair(gold_span, pred_span))

        self.recalculate_metrics_for_all_scorecards()

    def recalculate_metrics_for_all_scorecards(self) -> None:
        """Recalculates the metrics for all scorecards in the results aggregator.
        """
        for scoreCard in [self.strict_match, self.type_match, self.partial_match, self.bounds_match]:
            scoreCard.recalculate_metrics()