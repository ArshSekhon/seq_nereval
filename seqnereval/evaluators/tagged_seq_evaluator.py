from __future__ import annotations
from ..models import ResultAggregator, Span
from ..loaders import (
    TaggedSpanLoader, 
    IOBTaggedSpanLoader, 
    IOB2TaggedSpanLoader, 
    BIOESTaggedSpanLoader, 
    BILOUTaggedSpanLoader
)
from . import Evaluator


from collections import defaultdict
from typing import List, Tuple
from enum import Enum


class TaggedSeqEvaluator(Evaluator):
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
                 tokens: List[str],
                 gold_tags: List[str],
                 predicted_tags: List[str],
                 format_of_tags: TaggedSeqEvaluator.SupportedFormats,
                 context_padding=0):
        """Constructor for TaggedSeqEvaluator
        Args:
            tokens (List[str]): Lists of tokens.
            gold_tags (List[str]): Lists of golden tags.
            predicted_tags (List[str]): Lists of predicted tags.
            format_of_tags (TaggedSeqEvaluator.SupportedFormats): Format of the tagged sequence.
            context_padding (int): Number of tokens on each side of span to consider as part of context for error analysis.
        """

        self.tokens = list(tokens)
        self.gold_tags = list(gold_tags)
        self.predicted_tags = list(predicted_tags)
        self.context_padding = context_padding
        self.format_of_tags = format_of_tags

        gold_spans = self.__convert_tags_grouped_by_docs_to_spans_grouped_by_docs(self.tokens, self.gold_tags)
        predicted_spans = self.__convert_tags_grouped_by_docs_to_spans_grouped_by_docs(self.tokens, self.predicted_tags)

        super().__init__(gold_spans, predicted_spans)


    def __convert_tags_grouped_by_docs_to_spans_grouped_by_docs(self, 
                                                                tokens: List[str], 
                                                                tags: List[str]):
        Loader  = TaggedSeqEvaluator.loader_mapping[self.format_of_tags]
        loader = Loader(tokens, tags, self.context_padding)
        return loader.retreive_spans()
    