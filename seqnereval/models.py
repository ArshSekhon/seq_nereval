from __future__ import annotations
from copy import deepcopy
from typing import List, Tuple, Dict


class NEREntitySpan:
    def __init__(self, entity_type: str, start_idx: int, end_idx: int, tokens_spanned: List[str] = [], span_context=None):
        """
        Constructor for NEREntitySpan

        Parameters:
            entity_type (str): type/label of entity.
            start_idx (int): index of the first token that is a part of the entity.
            end_idx (int): index of the last token that is a part of the entity.
        """

        self.entity_type = entity_type
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.tokens_spanned = tokens_spanned
        if span_context == None:
            self.span_context = self.tokens_spanned
        else:
            self.span_context = span_context

    def __str__(self):
        return (f'Entity Type: "{self.entity_type}", Span:({self.start_idx},'
                f' {self.end_idx}), Tokens:{self.tokens_spanned}')

    def __repr__(self):
        return (f'Entity Type: "{self.entity_type}", Span:({self.start_idx},'
                f' {self.end_idx}), Tokens:{self.tokens_spanned}')

    def __hash__(self):
        return hash(f'{self.entity_type}-{self.start_idx}-{self.end_idx}')

    def __eq__(self, other):
        return (self.entity_type == other.entity_type and
                self.start_idx == other.start_idx and
                self.end_idx == other.end_idx)

    def spans_same_tokens_as(self, otherEntity):
        """
        Finds if both the entities span same tokens regardless of type.

        Parameters:
            otherEntity (NEREntitySpan): other entity to check
        Returns:
            'True' if they both span the same tokens, else 'False'
        """

        return (self.start_idx == otherEntity.start_idx and self.end_idx == otherEntity.end_idx)

    def overlaps_with(self, otherEntity) -> bool:
        """
        Finds if the given span overlaps

        Parameters:
            otherEntity (NEREntitySpan): other span to check for overlap
        Returns:
            'True' if there is an overlap, else 'False'
        """

        return max(self.start_idx, otherEntity.start_idx) <= min(self.end_idx, otherEntity.end_idx)


class NERResultAggregator:
    def __init__(self):
        """
        Constructor for NERResult 
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

        self.type_match_span_match: List[Tuple[NEREntitySpan, NEREntitySpan]] = [
        ]
        self.unecessary_predicted_entity: List[NEREntitySpan] = []
        self.missed_gold_entity: List[NEREntitySpan] = []
        self.type_mismatch_span_match: List[Tuple[NEREntitySpan, NEREntitySpan]] = [
        ]
        self.type_match_span_partial: List[Tuple[NEREntitySpan, NEREntitySpan]] = [
        ]
        self.type_mismatch_span_partial: List[Tuple[NEREntitySpan, NEREntitySpan]] = [
        ]

    def append_result_aggregator(self, otherResults: NERResultAggregator) -> None:
        """Appends the results obtained from a different evaluation.

        Args:
            otherResults (NERResult): Result to be appended.
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
            [self.type_match_span_match, self.unecessary_predicted_entity, self.missed_gold_entity,
                self.type_mismatch_span_match, self.type_match_span_partial, self.type_mismatch_span_partial],
            [otherResults.type_match_span_match, otherResults.unecessary_predicted_entity, otherResults.missed_gold_entity,
                otherResults.type_mismatch_span_match, otherResults.type_match_span_partial, otherResults.type_mismatch_span_partial]
        ):
            ownScenarios.extend(otherScenarios)

        self.refresh_metrics()

    # Scenario I
    def add_type_match_span_match(self, gold_entity: NEREntitySpan, pred_entity: NEREntitySpan) -> None:
        """Add Gold and Predicted Entity span pair to Scenario I aggregator: both type and span match.

        Args:
            gold_entity (NEREntitySpan): Golden Entity Span.
            pred_entity (NEREntitySpan): Predicted Entity Span.
        """

        self.type_match_span_match.append((gold_entity, pred_entity))

        self.strict_match["correct"].append((gold_entity, pred_entity))
        self.type_match["correct"].append((gold_entity, pred_entity))
        self.partial_match["correct"].append((gold_entity, pred_entity))
        self.bounds_match["correct"].append((gold_entity, pred_entity))

        self.refresh_metrics()

    # Scenario II

    def add_unecessary_predicted_entity(self, uncessary_pred_entity: NEREntitySpan) -> None:
        """Add wrongly predicted entity span to the Scenario II aggregate: predicted 
            entity span doesn't exist in golden dataset.

        Args:
            uncessary_pred_entity (NEREntitySpan): Entity span that was wrongly predicted.
        """

        self.unecessary_predicted_entity.append(uncessary_pred_entity)

        self.strict_match["spurious"].append(uncessary_pred_entity)
        self.type_match["spurious"].append(uncessary_pred_entity)
        self.partial_match["spurious"].append(uncessary_pred_entity)
        self.bounds_match["spurious"].append(uncessary_pred_entity)

        self.refresh_metrics()

    # Scenario III
    def add_missed_gold_entity(self, missed_gold_entity: NEREntitySpan) -> None:
        """Add missed entity span to Scenario III aggregate: missed entity span that 
            wasn't predicted

        Args:
            missed_gold_entity (NEREntitySpan): Entity span that wasn't predicted.
        """

        self.missed_gold_entity.append(missed_gold_entity)

        self.strict_match["missed"].append(missed_gold_entity)
        self.type_match["missed"].append(missed_gold_entity)
        self.partial_match["missed"].append(missed_gold_entity)
        self.bounds_match["missed"].append(missed_gold_entity)

        self.refresh_metrics()

    # Scenario IV
    def add_type_mismatch_span_match(self, gold_entity: NEREntitySpan, pred_entity: NEREntitySpan) -> None:
        """Add Gold and predicted entity pair for which span was predicted correctly but the 
            entity type was incorrect.

        Args:
            gold_entity (NEREntitySpan): Gold entity span for which span was predicted
                                            correctly but the type wasn't.
            pred_entity (NEREntitySpan): Predicted Entity span for which span was correct
                                            but the type was not.
        """

        self.type_mismatch_span_match.append((gold_entity, pred_entity))

        self.strict_match["incorrect"].append((gold_entity, pred_entity))
        self.type_match["incorrect"].append((gold_entity, pred_entity))
        self.partial_match["correct"].append((gold_entity, pred_entity))
        self.bounds_match["correct"].append((gold_entity, pred_entity))

        self.refresh_metrics()

    # Scenario V
    def add_type_match_span_partial(self, gold_entity: NEREntitySpan, pred_entity: NEREntitySpan) -> None:
        """Add Gold and predicted entity pair for which span was predicted partially correctly whereas the 
            entity type was predicted correctly. 

        Args:
            gold_entity (NEREntitySpan): Gold entity span for which span was predicted partially correctly
                whereas the entity type was predicted correctly. 
            pred_entity (NEREntitySpan): Predicted Entity span for which span was correct but the 
                type was not.
        """

        self.type_match_span_partial.append((gold_entity, pred_entity))

        self.strict_match["incorrect"].append((gold_entity, pred_entity))
        self.type_match["correct"].append((gold_entity, pred_entity))
        self.partial_match["partial"].append((gold_entity, pred_entity))
        self.bounds_match["incorrect"].append((gold_entity, pred_entity))

        self.refresh_metrics()

    # Scenario VI
    def add_type_mismatch_span_partial(self, gold_entity: NEREntitySpan, pred_entity: NEREntitySpan) -> None:
        """Add Gold and predicted entity pair for which span was predicted partially correctly and the entity
                type was incorrect.

        Args:
            gold_entity (NEREntitySpan): [description]
            pred_entity (NEREntitySpan): [description]
        """

        self.type_mismatch_span_partial.append((gold_entity, pred_entity))

        self.strict_match["incorrect"].append((gold_entity, pred_entity))
        self.type_match["incorrect"].append((gold_entity, pred_entity))
        self.partial_match["partial"].append((gold_entity, pred_entity))
        self.bounds_match["incorrect"].append((gold_entity, pred_entity))

        self.refresh_metrics()

    def refresh_metrics(self):
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
