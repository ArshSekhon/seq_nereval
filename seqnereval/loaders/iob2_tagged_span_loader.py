from __future__ import annotations
from typing import List
from . import IOBTaggedSpanLoader
from ..models import Span

class IOB2TaggedSpanLoader(IOBTaggedSpanLoader):
    def __init__(self, tokens: List[str], tags: List[str], context_padding:int = 2):
        super().__init__(tokens, tags, context_padding)

        self.start_prefix: str = 'B'
        self.inside_prefix: str = 'I'
        self.outside_tag: str = 'O'

        self.valid_token_tag_prefix = set([self.start_prefix, 
                                            self.inside_prefix, 
                                            self.outside_tag])
    
    def retreive_spans(self)->List[Span]: 
        self.reset_spans_aggregator()

        for idx, token_tag in enumerate(self.tags):
            if token_tag == self.outside_tag:
                if self.is_span_under_construction():
                    self.close_span_under_construction_and_save_span(idx)
                else: 
                    continue
            else:
                token_tag_prefix: str = self.get_prefix_for_tag_at(idx)

                if token_tag_prefix == self.start_prefix:
                    if self.is_span_under_construction():
                        self.close_span_under_construction_and_save_span(idx)
                    
                    self.initiate_construction_of_new_span(idx)

                elif token_tag_prefix == self.inside_prefix:
                    if self.is_span_under_construction():
                        continue
                    else:
                        raise Exception(f'Encountered inside tag before a start tag at idx: "{idx}".')

                elif token_tag_prefix not in self.valid_token_tag_prefix:
                    raise Exception(f'Unknown Token Tag: "{token_tag}" with prefix label: "{token_tag_prefix}"')

        if self.is_span_under_construction():
            self.close_span_under_construction_and_save_span(len(self.tags))
        
        return self.spans
