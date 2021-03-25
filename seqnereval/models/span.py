from __future__ import annotations
from typing import List

class Span:
    def __init__(self,
                 span_type: str,
                 start_idx: int, 
                 end_idx: int, 
                 spanned_tokens: List[str] = None, 
                 span_context: List[str] = None):
        """
        Construct a new Span.

        Parameters:
            type (str): type/label of span.
            start_idx (int): index of the first token that is a part of the span.
            end_idx (int): index of the last token that is a part of the span.
            spanned_tokens [optional, default = []] (List[str]): list of tokens spanned by the span.
            span_context [optional, default = spanned_tokens] (List[str]): list of tokens spanned by the span + 
                                        some surrounding tokens for context.
        """

        self.span_type = span_type 

        self.__validate_start_end(start_idx, end_idx)
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.spanned_tokens = spanned_tokens
        if span_context == None:
            self.span_context = self.spanned_tokens
        else:
            self.span_context = span_context

    def __str__(self):
        return (f'(Type: "{self.span_type}", Token Span IDX:({self.start_idx},'
                f' {self.end_idx}), Tokens:{self.spanned_tokens}, Context:{self.span_context})')

    def __repr__(self):
        return (f'(Type: "{self.span_type}", Token Span IDX:({self.start_idx},'
                f' {self.end_idx}), Tokens:{self.spanned_tokens}, Context:{self.span_context})')

    def __hash__(self):
        return hash(f'{self.span_type}-{self.start_idx}-{self.end_idx}')

    def __eq__(self, other):
        return (self.span_type == other.span_type and
                self.start_idx == other.start_idx and
                self.end_idx == other.end_idx)

    def bounds_same_tokens_as(self, otherSpan):
        """
        Finds if the other span has same bounds regardless of type.

        Parameters:
            otherSpan (Span): other span to check
        Returns:
            'True' if it has same bounds, else 'False'
        """

        return (self.start_idx == otherSpan.start_idx and self.end_idx == otherSpan.end_idx)

    def overlaps_with(self, otherSpan) -> bool:
        """
        Finds if the given other span overlaps this span.

        Parameters:
            span (Span): other span to check for overlap
        Returns:
            'True' if there is an overlap, else 'False'
        """

        return max(self.start_idx, otherSpan.start_idx) <= min(self.end_idx, otherSpan.end_idx)
    
    def ends_after_end_of(self, otherSpan):
        return self.end_idx > otherSpan.end_idx

    def starts_before_start_of(self, otherSpan):
        return self.start_idx < otherSpan.start_idx

    def starts_after_end_of(self, otherSpan):
        return self.start_idx > otherSpan.end_idx

    def ends_before_start_of(self, otherSpan):
        return self.end_idx < otherSpan.start_idx

    def __validate_start_end(self, start_idx, end_idx)->None:
        if start_idx>end_idx:
            raise Exception('Start IDX for a span cannot be > End IDX.')