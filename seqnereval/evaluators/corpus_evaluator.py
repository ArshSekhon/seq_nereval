from ..models import ResultAggregator, Span
from . import DocumentEvaluator

from collections import defaultdict
from typing import List, Tuple, Dict
from copy import deepcopy

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

        self.__result = None
        self.__results_grouped_by_tags = None
        self.__results_by_doc = None


    def evaluate(self) -> None:
        self.__reset_result_aggregators()

        for gold_spans, pred_spans in zip(self.gold_spans_grouped_by_doc, self.pred_spans_grouped_by_docs):
            document_evaluator = DocumentEvaluator(gold_spans, pred_spans)
            results_for_curr_doc, results_for_curr_doc_grouped_by_tags = document_evaluator.evaluate()

            self.__save_results_for_doc(results_for_curr_doc, results_for_curr_doc_grouped_by_tags)


    def get_result(self):
        if self.__result==None:
            raise Exception('Evaluation has not been performed yet. Please call evaluate() before retrieving results.')
        return self.__result

    def get_results_grouped_by_tags(self):
        if self.__results_grouped_by_tags==None:
            raise Exception('Evaluation has not been performed yet. Please call evaluate() before retrieving results.')
        return self.__results_grouped_by_tags

    def get_results_by_doc(self):
        if self.__results_by_doc==None:
            raise Exception('Evaluation has not been performed yet. Please call evaluate() before retrieving results.')
        return self.__results_by_doc


    def __reset_result_aggregators(self):
        self.__result = ResultAggregator()
        self.__results_grouped_by_tags = defaultdict(lambda: ResultAggregator())
        self.__results_by_doc = []

    def __save_results_for_doc(self, results_for_doc, results_grouped_by_tags_for_doc):
        self.__result.append_result_aggregator(results_for_doc)

        for key in results_grouped_by_tags_for_doc.keys():
            self.__results_grouped_by_tags[key].append_result_aggregator(results_grouped_by_tags_for_doc[key])

        self.__results_by_doc.append(results_for_doc)


    def __get_unique_span_types_from_spans_grouped_by_docs(self, spans_grouped_by_doc: List[List[Span]]) -> List[str]:
        return list(set([span.span_type
                            for spans in spans_grouped_by_doc
                                for span in spans]))
    