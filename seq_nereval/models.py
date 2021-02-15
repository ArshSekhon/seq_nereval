from copy import deepcopy
from typing import List, Tuple, Dict


class NEREntitySpan:
    def __init__(self, entity_type: str, start_idx: int, end_idx: int):
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

    def __str__(self):
        return f'"{self.entity_type}" ({self.start_idx}, {self.end_idx})'

    def __repr__(self):
        return f'"{self.entity_type}" ({self.start_idx}, {self.end_idx})'

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


class NERResult:
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
            "possible": [],
            "actual": [],
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

        self.strict_match = deepcopy(self.__metrics_template)
        self.type_match = deepcopy(self.__metrics_template)
        self.partial_match = deepcopy(self.__metrics_template)
        self.bounds_match = deepcopy(self.__metrics_template)

        self.type_match_span_match: List[Tuple[NEREntitySpan, NEREntitySpan]]=[]
        self.unecessary_predicted_entity: List[NEREntitySpan]=[]
        self.missed_gold_entity: List[NEREntitySpan] = []
        self.type_mismatch_span_match: List[Tuple[NEREntitySpan, NEREntitySpan]]=[]
        self.type_match_span_partial: List[Tuple[NEREntitySpan, NEREntitySpan]]=[]
        self.type_mismatch_span_partial: List[Tuple[NEREntitySpan, NEREntitySpan]] =[]

    def append_results(self, otherResults):
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

                if type(ownMetric) is int and type(otherResultMetric) is int:
                    ownMetric += otherResultMetric

        # Merge Scenarios
        for ownScenarios, otherScenarios in zip(
            [self.type_match_span_match, self.unecessary_predicted_entity, self.missed_gold_entity,
                self.type_mismatch_span_match, self.type_match_span_match, self.type_mismatch_span_partial],
            [otherResults.type_match_span_match, otherResults.unecessary_predicted_entity, otherResults.missed_gold_entity,
                otherResults.type_mismatch_span_match, otherResults.type_match_span_match, otherResults.type_mismatch_span_partial]
        ):
            ownScenarios.extend(otherScenarios)

        self.__update_metrics()

    # Scenario I
    def add_type_match_span_match(self, gold_entity: NEREntitySpan, pred_entity: NEREntitySpan):
        self.type_match_span_match.append((gold_entity, pred_entity))

        self.strict_match["correct"].append((gold_entity, pred_entity))
        self.type_match["correct"].append((gold_entity, pred_entity))
        self.partial_match["correct"].append((gold_entity, pred_entity))
        self.bounds_match["correct"].append((gold_entity, pred_entity))

        self.__update_metrics()

    # Scenario II

    def add_unecessary_predicted_entity(self, uncessary_pred_entity: NEREntitySpan):
        self.unecessary_predicted_entity.append(uncessary_pred_entity)

        self.strict_match["spurious"].append(uncessary_pred_entity)
        self.type_match["spurious"].append(uncessary_pred_entity)
        self.partial_match["spurious"].append(uncessary_pred_entity)
        self.bounds_match["spurious"].append(uncessary_pred_entity)

        self.__update_metrics()

    # Scenario III
    def add_missed_gold_entity(self, missed_gold_entity: NEREntitySpan):
        self.missed_gold_entity.append(missed_gold_entity)

        self.strict_match["missed"].append(missed_gold_entity)
        self.type_match["missed"].append(missed_gold_entity)
        self.partial_match["missed"].append(missed_gold_entity)
        self.bounds_match["missed"].append(missed_gold_entity)

        self.__update_metrics()

    # Scenario IV
    def add_type_mismatch_span_match(self, gold_entity: NEREntitySpan, pred_entity: NEREntitySpan):
        self.type_mismatch_span_match.append((gold_entity, pred_entity))

        self.strict_match["incorrect"].append((gold_entity, pred_entity))
        self.type_match["incorrect"].append((gold_entity, pred_entity))
        self.partial_match["correct"].append((gold_entity, pred_entity))
        self.bounds_match["correct"].append((gold_entity, pred_entity))

        self.__update_metrics()

    # Scenario V
    def add_type_match_span_partial(self, gold_entity: NEREntitySpan, pred_entity: NEREntitySpan):
        self.type_match_span_partial.append((gold_entity, pred_entity))

        self.strict_match["incorrect"].append((gold_entity, pred_entity))
        self.type_match["correct"].append((gold_entity, pred_entity))
        self.partial_match["partial"].append((gold_entity, pred_entity))
        self.bounds_match["incorrect"].append((gold_entity, pred_entity))

        self.__update_metrics()

    # Scenario VI
    def add_type_mismatch_span_partial(self, gold_entity: NEREntitySpan, pred_entity: NEREntitySpan):
        self.type_mismatch_span_partial.append((gold_entity, pred_entity))

        self.strict_match["incorrect"].append((gold_entity, pred_entity))
        self.type_match["incorrect"].append((gold_entity, pred_entity))
        self.partial_match["partial"].append((gold_entity, pred_entity))
        self.bounds_match["incorrect"].append((gold_entity, pred_entity))

        self.__update_metrics()

    def __update_metrics(self):
        for result_scheme in [self.strict_match, self.type_match,
                              self.partial_match, self.bounds_match]:
            self.__compute_actual_possible(result_scheme)
            self.__compute_precision_recall(result_scheme)

    def __compute_actual_possible(self, results):
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
