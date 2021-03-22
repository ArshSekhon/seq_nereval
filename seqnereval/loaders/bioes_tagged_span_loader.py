from __future__ import annotations
from typing import List
from . import IOBTaggedSpanLoader
from ..models import Span

class BIOESTaggedSpanLoader(IOBTaggedSpanLoader):
    def __init__(self, tokens: List[str], tags: List[str], context_padding:int = 2):
        super().__init__(tokens, tags, context_padding)

        self.start_prefix: str = 'B'
        self.inside_prefix: str = 'I'
        self.outside_tag: str = 'O'
        self.end_prefix: str = 'E'
        self.single_prefix: str = 'S'


        self.valid_token_tag_prefix = set([self.start_prefix, 
                                            self.inside_prefix, 
                                            self.outside_tag, 
                                            self.end_prefix, 
                                            self.single_prefix])
    
    def retreive_spans(self)->List[Span]:
        self.reset_spans_aggregator()

        for idx, token_tag in enumerate(self.tags):
            if token_tag == self.outside_tag:
                if self.is_span_under_construction():
                    raise Exception(f'Encountered an outside tag before a span was closed at idx: {idx}.')
                else:
                    continue
            else:
                token_tag_prefix: str = self.get_prefix_for_tag_at(idx)

                if token_tag_prefix == self.start_prefix:
                    if self.is_span_under_construction():
                        raise Exception(f'Encountered a start tag before a span was closed at idx: {idx}.')
                    else:
                        self.initiate_construction_of_new_span(idx)

                elif token_tag_prefix == self.single_prefix:
                    if self.is_span_under_construction():
                        raise Exception(f'Encountered a single start tag before a span was closed at idx: {idx}.')
                    else:
                        self.initiate_construction_of_new_span(idx)
                        self.close_span_under_construction_and_save_span(idx+1)
                        
                elif token_tag_prefix == self.inside_prefix:
                    if self.is_span_under_construction():
                        continue
                    else:
                        raise Exception(f'Encountered an inside tag before a start tag at idx: {idx}.')
                
                elif token_tag_prefix == self.end_prefix:
                    if self.is_span_under_construction():
                        self.close_span_under_construction_and_save_span(idx+1)
                    else:
                        raise Exception(f'Encountered an end tag before a start tag at idx: {idx}.')

                elif token_tag_prefix not in self.valid_token_tag_prefix:
                    raise Exception(f'Unknown Token Tag: "{token_tag}" with an unknown prefix label: "{token_tag_prefix}" at idx: {idx}.')

        if self.is_span_under_construction():
            raise Exception(f'Tag List ended before a span was closed at the idx: {len(self.tags)-1}')
        
        return self.spans
