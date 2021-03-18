from __future__ import annotations
from copy import deepcopy
from typing import List, Tuple, Dict
from . import Span, GoldPredictedPair

class ResultAggregator:
    def __init__(self):
        """
        Constructor for ResultAggregator 
        """
        self.__metrics_template = {
            "correct": [],
            "incorrect": [],
            "partial": [],
            "missed": [],
            "spurious": [],
            "possible": 0,
            "actual": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

        self.strict_match = deepcopy(self.__metrics_template)
        self.type_match = deepcopy(self.__metrics_template)
        self.partial_match = deepcopy(self.__metrics_template)
        self.bounds_match = deepcopy(self.__metrics_template)

        self.type_match_bounds_match: List[GoldPredictedPair] = []
        self.unecessary_predicted_span: List[Span] = []
        self.missed_gold_span: List[Span] = []
        self.type_mismatch_bounds_match: List[GoldPredictedPair] = []
        self.type_match_bounds_partial: List[GoldPredictedPair] = []
        self.type_mismatch_bounds_partial: List[GoldPredictedPair] = []

    def summarize_result(self):
        """Summarizes the results into numbers.
        """
        def summarize_metric(metric):
            summary = metric.copy()
            for key in metric.keys():
                if type(metric[key]) is list:
                    summary[key] = len(metric[key])
            return summary

        return {
            "strict_match": summarize_metric(self.strict_match),
            "type_match": summarize_metric(self.type_match),
            "partial_match": summarize_metric(self.partial_match),
            "bounds_match": summarize_metric(self.bounds_match),
            "type_match_bounds_match": len(self.type_match_bounds_match),
            "unecessary_predicted_span": len(self.unecessary_predicted_span),
            "missed_gold_span": len(self.missed_gold_span),
            "type_mismatch_bounds_match": len(self.type_mismatch_bounds_match),
            "type_match_bounds_partial": len(self.type_match_bounds_partial),
            "type_mismatch_bounds_partial": len(self.type_mismatch_bounds_partial)
        }

    def append_result_aggregator(self, otherResults: ResultAggregator) -> None:
        """Appends the results obtained from a different evaluation.

        Args:
            otherResults (ResultAggregator): Result to be appended.
        """

        for ownResultScheme, otherResultScheme in zip(
            [self.strict_match, self.type_match,
                self.partial_match, self.bounds_match],
            [otherResults.strict_match, otherResults.type_match,
                otherResults.partial_match, otherResults.bounds_match]
        ):
            for metric_key in ownResultScheme.keys():
                ownMetric = ownResultScheme[metric_key]
                otherResultMetric = otherResultScheme[metric_key]

                if type(ownMetric) is list and type(otherResultMetric) is list:
                    ownMetric.extend(otherResultMetric)

        # Merge Scenarios
        for ownScenarios, otherScenarios in zip(
            [self.type_match_bounds_match, self.unecessary_predicted_span, self.missed_gold_span,
                self.type_mismatch_bounds_match, self.type_match_bounds_partial, self.type_mismatch_bounds_partial],
            [otherResults.type_match_bounds_match, otherResults.unecessary_predicted_span, otherResults.missed_gold_span,
                otherResults.type_mismatch_bounds_match, otherResults.type_match_bounds_partial, otherResults.type_mismatch_bounds_partial]
        ):
            ownScenarios.extend(otherScenarios)

        self.recalculate_metrics()

    # Scenario I
    def add_type_match_bounds_match(self, gold_span: Span, pred_span: Span) -> None:
        """Add Gold and Predicted span pair to Scenario I aggregator: both type and bounds match.

        Args:
            gold_span (Span): Golden Span.
            pred_span (Span): Predicted Span.
        """

        self.type_match_bounds_match.append(
            GoldPredictedPair(gold_span, pred_span))

        self.strict_match["correct"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.type_match["correct"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.partial_match["correct"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.bounds_match["correct"].append(
            GoldPredictedPair(gold_span, pred_span))

        self.recalculate_metrics()

    # Scenario II

    def add_unecessary_predicted_span(self, uncessary_pred_span: Span) -> None:
        """Add wrongly predicted span to the Scenario II aggregate: predicted 
           span doesn't exist in golden dataset.

        Args:
            uncessary_pred_span (Span): Span that was wrongly predicted.
        """

        self.unecessary_predicted_span.append(uncessary_pred_span)

        self.strict_match["spurious"].append(uncessary_pred_span)
        self.type_match["spurious"].append(uncessary_pred_span)
        self.partial_match["spurious"].append(uncessary_pred_span)
        self.bounds_match["spurious"].append(uncessary_pred_span)

        self.recalculate_metrics()

    # Scenario III
    def add_missed_gold_span(self, missed_gold_span: Span) -> None:
        """Add missed span to Scenario III aggregate: missed span that 
            wasn't predicted

        Args:
            missed_gold_span (Span): Span that wasn't predicted.
        """

        self.missed_gold_span.append(missed_gold_span)

        self.strict_match["missed"].append(missed_gold_span)
        self.type_match["missed"].append(missed_gold_span)
        self.partial_match["missed"].append(missed_gold_span)
        self.bounds_match["missed"].append(missed_gold_span)

        self.recalculate_metrics()

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

        self.type_mismatch_bounds_match.append(
            GoldPredictedPair(gold_span, pred_span))

        self.strict_match["incorrect"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.type_match["incorrect"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.partial_match["correct"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.bounds_match["correct"].append(
            GoldPredictedPair(gold_span, pred_span))

        self.recalculate_metrics()

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

        self.type_match_bounds_partial.append(
            GoldPredictedPair(gold_span, pred_span))

        self.strict_match["incorrect"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.type_match["correct"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.partial_match["partial"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.bounds_match["incorrect"].append(
            GoldPredictedPair(gold_span, pred_span))

        self.recalculate_metrics()

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

        self.strict_match["incorrect"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.type_match["incorrect"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.partial_match["partial"].append(
            GoldPredictedPair(gold_span, pred_span))
        self.bounds_match["incorrect"].append(
            GoldPredictedPair(gold_span, pred_span))

        self.recalculate_metrics()

    def recalculate_metrics(self):
        """Recalculates the metrics for the results aggregator.
        """

        for result_scheme in [self.strict_match, self.bounds_match]:
            self.__compute_actual_possible(result_scheme)
            self.__compute_precision_recall(result_scheme, False)

        for result_scheme in [self.type_match, self.partial_match]:
            self.__compute_actual_possible(result_scheme)
            self.__compute_precision_recall(result_scheme, True)

    def __compute_actual_possible(self, results):
        """Calculates the number of the actual and possible 

        Args:
            results: the results aggregator.

        Returns:
            result dictionary with updated values of actual and possible counts.
        """

        correct = len(results["correct"])
        incorrect = len(results["incorrect"])
        partial = len(results["partial"])
        missed = len(results["missed"])
        spurious = len(results["spurious"])

        possible = correct + incorrect + partial + missed
        actual = correct + incorrect + partial + spurious

        results["actual"] = actual
        results["possible"] = possible

        return results

    def __compute_precision_recall(self, results, partial_or_type=False):
        """Compute precision and recall for the results dictionary.

        Args:
            results: dictionary containing the results
            partial_or_type (bool, optional): [description]. Defaults to False.

        Returns:
            result dictionary with updated values of precision, recall and f1 score
                using the result aggregators.
        """

        actual = results["actual"]
        possible = results["possible"]
        partial = len(results["partial"])
        correct = len(results["correct"])

        if partial_or_type:
            precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
            recall = (correct + 0.5 * partial) / \
                possible if possible > 0 else 0

        else:
            precision = correct / actual if actual > 0 else 0
            recall = correct / possible if possible > 0 else 0

        results["precision"] = precision
        results["recall"] = recall
        results["f1"] = (
            2 * (precision * recall) / (precision +
                                        recall) if (precision + recall) > 0 else 0
        )

        return results
