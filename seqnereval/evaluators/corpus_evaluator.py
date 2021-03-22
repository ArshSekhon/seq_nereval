from ..models import ResultAggregator, Span
from collections import defaultdict
from typing import List, Tuple

class CorpusEvaluator:
    def __init__(self, gold_spans_grouped_by_doc: List[List[Span]], pred_spans_grouped_by_doc: List[List[Span]]):
        """
        Construct a LabelledSpanEvaluator

        Args:
            gold_entity_spans_grouped_by_doc (List[List[Span]]): List of gold spans grouped by documents.
            pred_spans_grouped_by_doc (List[List[Span]]): List of predicted spans grouped by documents.
        """

        if len(gold_spans_grouped_by_doc) != len(pred_spans_grouped_by_doc):
            raise Exception(f'# of documents for which golden tags were provided {len(gold_spans_grouped_by_doc)}'
                            f'!= # of documents for which predicted tags were provided {len(pred_spans_grouped_by_doc)}')

        self.gold_spans_grouped_by_doc = gold_spans_grouped_by_doc
        self.pred_spans_grouped_by_docs = pred_spans_grouped_by_doc
        
        self.unique_gold_span_types = self.__get_unique_span_types_from_spans_grouped_by_docs(gold_spans_grouped_by_doc)

        self.results = ResultAggregator()
        self.results_grouped_by_tags = defaultdict(lambda: ResultAggregator())



    def evaluate(self) -> Tuple[ResultAggregator, ResultAggregator]:
        """Runs the evaluation and return results

        Returns:
            Tuple[ResultAggregator, ResultAggregator]: (Results, Results Grouped by tags)
        """
        results_by_doc = []
        results = ResultAggregator()

        for gold_spans, pred_spans in zip(self.gold_spans_grouped_by_doc, self.pred_spans_grouped_by_docs):
            results_for_curr_doc = self.__evaluate_results_for_single_doc(gold_spans, pred_spans)
            results_by_doc.append(results_for_curr_doc)
            results.append_result_aggregator(results_for_curr_doc[0])

        return results, results_for_curr_doc



    def __evaluate_results_for_single_doc(self, gold_spans: List[Span], predicted_spans: List[Span]) -> Tuple[ResultAggregator, ResultAggregator]:
        """Calculate the metrics for a particular document.

        Args:
            gold_spans (List[Span]): List of gold labelled spans
            predicted_spans (List[Span]): List of predicted labelled spans

        Returns:
            Tuple[ResultAggregator, ResultAggregator]: (Results, Results Grouped by tags)
        """

        def entity_span_sort_fn(span): return (span.start_idx, span.end_idx)

        # sort the entity list so we can make the evaluation faster (O(n)).
        gold_spans.sort(key=entity_span_sort_fn)
        predicted_spans.sort(key=entity_span_sort_fn)

        # to check if the gold span or pred span was overlapping in last step
        gold_part_overlap_in_last_step, pred_part_overlap_in_last_step = False, False

        gold_idx, pred_idx = 0, 0
        results = ResultAggregator()
        results_grouped_by_tags = defaultdict(lambda: ResultAggregator())

        while gold_idx < len(gold_spans) and pred_idx < len(predicted_spans):
            if gold_spans[gold_idx] == predicted_spans[pred_idx]:
                # Scenario I: Both entity type/labels and spans match perfectly
                results.add_type_match_bounds_match(
                    gold_spans[gold_idx], predicted_spans[pred_idx])

                gold_span_type = gold_spans[gold_idx].span_type
                results_grouped_by_tags[gold_span_type].add_type_match_bounds_match(
                    gold_spans[gold_idx],
                    predicted_spans[pred_idx]
                )

                # it is safe to move cursor over
                # as overlapping spans are not allowed within the predicted entity spans list
                # and is also not allowed within gold entity span list
                gold_idx += 1
                pred_idx += 1

                gold_part_overlap_in_last_step, pred_part_overlap_in_last_step = False, False

            elif gold_spans[gold_idx].bounds_same_tokens_as(predicted_spans[pred_idx]):
                # Scenario IV: Wrong Entity types but, spans match perfectly
                results.add_type_mismatch_bounds_match(
                    gold_spans[gold_idx], predicted_spans[pred_idx])

                gold_span_type = gold_spans[gold_idx].span_type
                results_grouped_by_tags[gold_span_type].add_type_mismatch_bounds_match(
                    gold_spans[gold_idx],
                    predicted_spans[pred_idx]
                )

                # it is safe to move cursor over
                # as overlapping spans are not allowed within the predicted entity spans list
                # and is also not allowed within gold entity span list
                gold_idx += 1
                pred_idx += 1

                gold_part_overlap_in_last_step, pred_part_overlap_in_last_step = False, False

            elif gold_spans[gold_idx].overlaps_with(predicted_spans[pred_idx]):
                if gold_spans[gold_idx].span_type == predicted_spans[pred_idx].span_type:
                    # Scenario V: Correct Entity Type, partial span overlap
                    results.add_type_match_bounds_partial(
                        gold_spans[gold_idx],
                        predicted_spans[pred_idx]
                    )

                    gold_span_type = gold_spans[gold_idx].span_type
                    results_grouped_by_tags[gold_span_type].add_type_match_bounds_partial(
                        gold_spans[gold_idx],
                        predicted_spans[pred_idx]
                    )
                else:
                    # Scenario VI: Wrong Entity Type, partial span overlap
                    results.add_type_mismatch_bounds_partial(
                        gold_spans[gold_idx],
                        predicted_spans[pred_idx]
                    )

                    gold_span_type = gold_spans[gold_idx].span_type
                    results_grouped_by_tags[gold_span_type].add_type_mismatch_bounds_partial(
                        gold_spans[gold_idx],
                        predicted_spans[pred_idx]
                    )

                if predicted_spans[pred_idx].end_idx > gold_spans[gold_idx].end_idx:
                    pred_part_overlap_in_last_step = True
                    gold_idx += 1
                elif predicted_spans[pred_idx].end_idx < gold_spans[gold_idx].end_idx:
                    gold_part_overlap_in_last_step = True
                    pred_idx += 1
                else:
                    gold_idx += 1
                    pred_idx += 1
                    gold_part_overlap_in_last_step, pred_part_overlap_in_last_step = False, False

            else:
                if predicted_spans[pred_idx].start_idx > gold_spans[gold_idx].end_idx:
                    if not gold_part_overlap_in_last_step:
                        # Scenario III system missed an entity
                        results.add_missed_gold_span(
                            gold_spans[gold_idx])

                        gold_span_type = gold_spans[gold_idx].span_type
                        results_grouped_by_tags[gold_span_type].add_missed_gold_span(
                            gold_spans[gold_idx]
                        )

                    gold_idx += 1
                    gold_part_overlap_in_last_step = False
                elif predicted_spans[pred_idx].end_idx < gold_spans[gold_idx].start_idx:
                    if not pred_part_overlap_in_last_step:
                        # Scenario II system hypothesised an extra entity
                        results.add_unecessary_predicted_span(
                            predicted_spans[pred_idx])

                        pred_span_type = predicted_spans[pred_idx].span_type
                        results_grouped_by_tags[pred_span_type].add_unecessary_predicted_span(
                            predicted_spans[pred_idx]
                        )

                    pred_idx += 1
                    pred_part_overlap_in_last_step = True

        if gold_part_overlap_in_last_step:
            gold_idx += 1

        while gold_idx < len(gold_spans):
            # Scenario III: missed entity
            results.add_missed_gold_span(
                gold_spans[gold_idx])

            gold_span_type = gold_spans[gold_idx].span_type
            results_grouped_by_tags[gold_span_type].add_missed_gold_span(
                gold_spans[gold_idx]
            )

            gold_idx += 1

        if pred_part_overlap_in_last_step:
            pred_idx += 1
        while pred_idx < len(predicted_spans):
            # Scenario II: hypothesised entity incorrect
            results.add_unecessary_predicted_span(
                predicted_spans[pred_idx])

            pred_span_type = predicted_spans[pred_idx].span_type
            results_grouped_by_tags[pred_span_type].add_unecessary_predicted_span(
                predicted_spans[pred_idx]
            )

            pred_idx += 1

        return results, results_grouped_by_tags

    def __get_unique_span_types_from_spans_grouped_by_docs(self, spans_grouped_by_doc: List[List[Span]]) -> List[str]:
        return list(set([span.span_type
                            for spans in spans_grouped_by_doc
                                for span in spans]))
    