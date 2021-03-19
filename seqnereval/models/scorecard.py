from __future__ import annotations
from typing import Dict
class ScoreCard:
    def __init__(self, is_partial_or_type_scorecard=False):
        self.correct = []
        self.incorrect = []
        self.partial = []
        self.missed = []
        self.spurious = []

        self.possible = 0
        self.actual = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        
        self.is_partial_or_type_scorecard = is_partial_or_type_scorecard

    def get_score_counts(self) -> Dict[str, int]:
        return {
            "correct_counts": len(self.correct),
            "incorrect_counts": len(self.incorrect),
            "partial_counts": len(self.partial),
            "missed_counts": len(self.missed),
            "spurious_counts": len(self.spurious)
        }

    def get_summary(self) -> Dict[str, int]:
        return {
            **self.get_score_counts(),
            "possible": 0,
            "actual": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

    def __compute_actual_possible(self) -> None:
        """Calculates the number of the actual and possible
        """
        scorecard_counts = self.get_score_counts()

        correct = scorecard_counts["correct_counts"]
        incorrect = scorecard_counts["incorrect_counts"]
        partial = scorecard_counts["partial_counts"]
        missed = scorecard_counts["missed_counts"]
        spurious = scorecard_counts["spurious_counts"]

        self.possible = correct + incorrect + partial + missed
        self.actual = correct + incorrect + partial + spurious

    def recalculate_metrics(self) -> None:
        """Compute precision and recall for the results dictionary.

        Args:
            partial_or_type (bool, optional): [description]. Defaults to False.
        """

        self.__compute_actual_possible()

        partial_count = len(self.partial)
        correct_count = len(self.correct)

        if self.is_partial_or_type_scorecard:
            self.precision = (correct_count + 0.5 * partial_count) / \
                self.actual if self.actual > 0 else 0
            self.recall = (correct_count + 0.5 * partial_count) / \
                self.possible if self.possible > 0 else 0

        else:
            self.precision = correct_count / self.actual if self.actual > 0 else 0
            self.recall = correct_count / self.possible if self.possible > 0 else 0

        self.f1 = (
            2 * (self.precision * self.recall) / (self.precision +
                                                  self.recall) if (self.precision + self.recall) > 0 else 0
        )

    def mergeScoreCard(self, scoreCardToMerge: ScoreCard) -> None:
        """Merges other scorecard into self.

        Args:
            scoreCardToMerge (ScoreCard)
        """
        self.correct.extend(scoreCardToMerge.correct)
        self.incorrect.extend(scoreCardToMerge.incorrect)
        self.partial.extend(scoreCardToMerge.partial)
        self.spurious.extend(scoreCardToMerge.spurious)
        self.missed.extend(scoreCardToMerge.missed)

        self.recalculate_metrics()

