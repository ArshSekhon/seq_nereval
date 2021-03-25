from __future__ import annotations
from ..models import ResultAggregator, Span
from ..loaders import TaggedSpanLoader, IOBTaggedSpanLoader, IOB2TaggedSpanLoader, BIOESTaggedSpanLoader, BILOUTaggedSpanLoader
from . import CorpusEvaluator


from collections import defaultdict
from typing import List, Tuple
from enum import Enum


class SeqTaggedCorpusEvaluator(CorpusEvaluator):
    class SupportedFormats(Enum):
        iob = 'IOB'
        iob2 = 'IOB2'
        bioes = 'BIOES'
        bilou = 'BILOU'

    loader_mapping = {
        SupportedFormats.iob: IOBTaggedSpanLoader,
        SupportedFormats.iob2: IOB2TaggedSpanLoader,
        SupportedFormats.bioes: BIOESTaggedSpanLoader,
        SupportedFormats.bioes: BILOUTaggedSpanLoader
    }

    def __init__(self,
                 tokens_grouped_by_docs: List[List[str]],
                 gold_tags_grouped_by_docs: List[List[str]],
                 predicted_tags_grouped_by_docs: List[List[str]],
                 format_of_tags: SeqTaggedCorpusEvaluator.SupportedFormats,
                 context_padding=0):
        """Constructor for SeqTaggedCorpusEvaluator

        Args:
            tokens_grouped_by_docs (List[List[str]]): Lists of tokens grouped by documents.
            gold_tags_grouped_by_docs (List[List[str]]): Lists of golden tags grouped by documents.
            predicted_tags_grouped_by_docs (List[List[str]]): Lists of predicted tags grouped by documents.
            format_of_tags (SeqLabelledSpanEvaluator.SupportedFormats): Format of the tagged sequence.
            context_padding (int): Number of tokens on each side of span to consider as part of context for error analysis.
        """

        if not self.__is_list_two_dimensional(tokens_grouped_by_docs):
            raise Exception('SeqTaggedCorpusEvaluator creation failed. tokens_grouped_by_docs is not a 2D List.'
                            ' Please pass a 2-D List into the constructor containing token spans grouped by docs.')

        if not self.__is_list_two_dimensional(gold_tags_grouped_by_docs):
            raise Exception('SeqTaggedCorpusEvaluator creation failed. gold_tags_grouped_by_docs is not a 2D List.'
                            ' Please pass a 2-D List into the constructor containing gold spans grouped by docs.')

        if not self.__is_list_two_dimensional(predicted_tags_grouped_by_docs):
            raise Exception('SeqTaggedCorpusEvaluator creation failed. predicted_tags_grouped_by_docs is not a 2D List.'
                            ' Please pass a 2-D List into the constructor containing predicted spans grouped by docs.')

        self.tokens_grouped_by_docs = list(tokens_grouped_by_docs)
        self.gold_tags_grouped_by_docs = list(gold_tags_grouped_by_docs)
        self.predicted_tags_grouped_by_docs = list(
            predicted_tags_grouped_by_docs)
        self.context_padding = context_padding
        self.format_of_tags = format_of_tags

        gold_spans_grouped_by_docs = self.__convert_tags_grouped_by_docs_to_spans_grouped_by_docs(
            self.gold_tags_grouped_by_docs, self.tokens_grouped_by_docs)
        predicted_spans_grouped_by_docs = self.__convert_tags_grouped_by_docs_to_spans_grouped_by_docs(
            self.predicted_tags_grouped_by_docs, self.tokens_grouped_by_docs)

        super().__init__(gold_spans_grouped_by_docs, predicted_spans_grouped_by_docs)

    def __convert_tags_grouped_by_docs_to_spans_grouped_by_docs(self, 
                                                                tags_grouped_by_docs: List[List[str]], 
                                                                tokens_grouped_by_docs: List[List[str]]):
        results = []
        Loader: TaggedSpanLoader = SeqTaggedCorpusEvaluator.loader_mapping[self.format_of_tags]
        for tags_for_a_doc, tokens_for_a_doc in zip(tags_grouped_by_docs, tokens_grouped_by_docs):
            loader = Loader(tags_for_a_doc, tokens_for_a_doc,
                            self.context_padding)
            results.append(loader.retreive_spans())
        return results

    def __is_list_two_dimensional(self, input_list):
        return self.__find_dimension_of_list(input_list) == 2

    def __find_dimension_of_list(self, input_list):

        def helper(input_list, dim=0):
            if isinstance(input_list, list):
                if input_list == []:
                    return dim
                dim = dim + 1
                dim = helper(input_list[0], dim)
                return dim
            else:
                if dim == 0:
                    return -1
                else:
                    return dim

        return helper(input_list)
