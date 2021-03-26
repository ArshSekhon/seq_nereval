from __future__ import annotations
from typing import List, Tuple, Dict
from collections import defaultdict

from ..models import Span, ResultAggregator

class Evaluator:
    def __init__(self, gold_spans: List[Span], predicted_spans: List[Span]):
        
        self.gold_spans = gold_spans
        self.predicted_spans = predicted_spans
        
        self.__validate_gold_and_predicted_spans()
        self.__sort_gold_and_predicted_spans()

        self.__result = None
        self.__results_grouped_by_tags = None


        # temporary variables to be used during evaluation
        self.__gold_spans_cursor = 0
        self.__predicted_spans_cursor = 0
        self.__gold_span_overlap_in_last_step = False
        self.__predicted_span_overlap_in_last_step = False



    def evaluate(self) -> Tuple[ResultAggregator, Dict[str,ResultAggregator]]:
        self.__reset_evaluator_state()

        while self.__gold_spans_cursor < len(self.gold_spans) and self.__predicted_spans_cursor < len(self.predicted_spans):
            current_gold_span = self.gold_spans[self.__gold_spans_cursor]
            current_predicted_span = self.predicted_spans[self.__predicted_spans_cursor]

            if current_gold_span == current_predicted_span:

                self.__report_type_match_bounds_match(current_gold_span, current_predicted_span)
                # it is safe to move cursors over as overlapping spans are not allowed 
                # in the predicted or gold spans list, so no future overlaps for current spans possible
                self.__gold_spans_cursor += 1
                self.__predicted_spans_cursor += 1

                self.__gold_span_overlap_in_last_step = False
                self.__predicted_span_overlap_in_last_step = False

            elif current_gold_span.bounds_same_tokens_as(current_predicted_span):
                
                self.__report_type_mismatch_bounds_match(current_gold_span, current_predicted_span)
                # it is safe to move cursors over as no future overlaps for current spans possible
                self.__gold_spans_cursor += 1
                self.__predicted_spans_cursor += 1

                self.__gold_span_overlap_in_last_step = False
                self.__predicted_span_overlap_in_last_step = False

            elif current_gold_span.overlaps_with(current_predicted_span):
                if current_gold_span.span_type == current_predicted_span.span_type:
                    self.__report_type_match_bounds_overlap(current_gold_span, current_predicted_span)
                else:
                    self.__report_type_mismatch_bounds_overlap(current_gold_span, current_predicted_span)

                if current_predicted_span.ends_after_end_of(current_gold_span):
                    # only increment gold cursor as there could be future overlaps for predicted span
                    self.__predicted_span_overlap_in_last_step = True
                    self.__gold_spans_cursor += 1

                elif current_gold_span.ends_after_end_of(current_predicted_span):
                    self.__gold_span_overlap_in_last_step = True
                    self.__predicted_spans_cursor += 1

                else:
                    # it is safe to move cursors over as no future overlaps for current spans possible
                    self.__gold_spans_cursor += 1
                    self.__predicted_spans_cursor += 1
                    self.__gold_span_overlap_in_last_step = False
                    self.__predicted_span_overlap_in_last_step = False

            elif not current_gold_span.overlaps_with(current_predicted_span):

                if current_predicted_span.starts_after_end_of(current_gold_span):

                    if not self.__gold_span_overlap_in_last_step:
                        self.__report_missed_span(current_gold_span)
                    
                    self.__gold_spans_cursor += 1
                    self.__gold_span_overlap_in_last_step = False
                
                elif current_predicted_span.ends_before_start_of(current_gold_span):
                    if not self.__predicted_span_overlap_in_last_step:
                        self.__report_unwanted_span(current_predicted_span)
                    
                    self.__predicted_spans_cursor += 1
                    self.__predicted_span_overlap_in_last_step = False

        # if any gold spans are left behind report them as missed
        self.__report_remaining_gold_spans_as_missed()
        # if any predicted spans are left behind repor them as unwanted
        self.__report_remaining_predicted_spans_as_unwanted()
    
    
    def get_result(self):
        if self.__result==None:
            raise Exception('Evaluation has not been performed yet. Please call evaluate() before retrieving results.')
        return self.__result

    def get_results_grouped_by_tags(self):
        if self.__results_grouped_by_tags==None:
            raise Exception('Evaluation has not been performed yet. Please call evaluate() before retrieving results.')
        return self.__results_grouped_by_tags

    def __report_type_match_bounds_match(self, gold: Span, predicted: Span) -> None:
        # Scenario I: Both entity type/labels and spans match perfectly
        self.__result.add_type_match_bounds_match(gold, predicted)
        self.__results_grouped_by_tags[gold.span_type].add_type_match_bounds_match(gold, predicted)
    
    def __report_unwanted_span(self, span: Span)->None:
        # Scenario II system hypothesised an extra entity
        self.__result.add_unecessary_predicted_span(span)
        self.__results_grouped_by_tags[span.span_type].add_unecessary_predicted_span(span)

    def __report_missed_span(self, span: Span)->None:
        # Scenario III system missed an entity
        self.__result.add_missed_gold_span(span)
        self.__results_grouped_by_tags[span.span_type].add_missed_gold_span(span)

    def __report_type_mismatch_bounds_match(self, gold: Span, predicted: Span)-> None:
        # Scenario IV: Wrong Entity types but, spans match perfectly
        self.__result.add_type_mismatch_bounds_match(gold, predicted)
        self.__results_grouped_by_tags[gold.span_type].add_type_mismatch_bounds_match(gold, predicted)
    
    def __report_type_match_bounds_overlap(self, gold: Span, predicted: Span) -> None:
        # Scenario V: Correct Entity Type, partial span overlap
        self.__result.add_type_match_bounds_partial(gold, predicted)
        self.__results_grouped_by_tags[gold.span_type].add_type_match_bounds_partial(gold, predicted)
    
    def __report_type_mismatch_bounds_overlap(self, gold: Span, predicted: Span) -> None:
        # Scenario VI: Wrong Entity Type, partial span overlap
        self.__result.add_type_mismatch_bounds_partial(gold, predicted)
        self.__results_grouped_by_tags[gold.span_type].add_type_mismatch_bounds_partial(gold, predicted)

    def __validate_gold_and_predicted_spans(self):
        if self.__do_any_spans_overlap(self.gold_spans):
            raise Exception("Overlapping Gold Spans found: Overlapping spans are not currently supported.")

        if self.__do_any_spans_overlap(self.predicted_spans):
            raise Exception("Overlapping Predicted Spans found: Overlapping spans are not currently supported.")

    def __report_remaining_gold_spans_as_missed(self):
        # if there was already an overlap with another predicted span, gold span can't be
        #  considered missed and hence should be ignored
        if self.__gold_span_overlap_in_last_step:
            self.__gold_spans_cursor += 1

        while self.__gold_spans_cursor < len(self.gold_spans):
            self.__report_missed_span(self.gold_spans[self.__gold_spans_cursor])
            self.__gold_spans_cursor += 1
    
    def __report_remaining_predicted_spans_as_unwanted(self):
        # if there was already an overlap with another gold span, predicted span can't be
        #  considered missed and hence should be ignored
        
        if self.__predicted_span_overlap_in_last_step:
            self.__predicted_spans_cursor += 1
        
        while self.__predicted_spans_cursor < len(self.predicted_spans):
            # Scenario II: hypothesised entity incorrect
            self.__report_unwanted_span(self.predicted_spans[self.__predicted_spans_cursor]) 
            self.__predicted_spans_cursor += 1

    def __sort_gold_and_predicted_spans(self):
        self.gold_spans.sort(key=lambda span: (span.start_idx, span.end_idx))
        self.predicted_spans.sort(key=lambda span: (span.start_idx, span.end_idx))
 
    def __do_any_spans_overlap(self, spans:List[Span]):
        spans = sorted(spans, key=lambda span: (span.start_idx, span.end_idx))
        prev_span_end = None
        
        for span in spans:
            if prev_span_end is not None and span.start_idx<=prev_span_end:
                return True
            prev_span_end = span.end_idx
        
        return False

    def __reset_evaluator_state(self):
        self.__predicted_spans_cursor = 0
        self.__gold_spans_cursor = 0

        self.__gold_span_overlap_in_last_step = False
        self.__predicted_span_overlap_in_last_step = False

        self.__result = ResultAggregator()
        self.__results_grouped_by_tags = defaultdict(lambda: ResultAggregator())
