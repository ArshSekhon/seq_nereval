from __future__ import annotations
from typing import List
from .tagged_span_loader import TaggedSpanLoader
from ..models import Span

class IOBTaggedSpanLoader(TaggedSpanLoader):
    def __init__(self, tokens: List[str], tags: List[str], context_padding:int = 2):
        super().__init__(tokens, tags, context_padding)

        if(len(tokens)!=len(tags)):
            raise Exception(f'Number of tokens and tags is not the same.')

        self.start_prefix: str = 'B'
        self.inside_prefix: str = 'I'
        self.outside_tag: str = 'O'

        self.valid_token_tag_prefix = set([self.start_prefix, 
                                            self.inside_prefix, 
                                            self.outside_tag])
        self.spans: List[str] = None
        self.open_span_under_construction: Span = None
    
    def retreive_spans(self)->List[Span]:
        self.reset_spans_aggregator()

        for idx, token_tag in enumerate(self.tags):
            token_tag_prefix: str = self.get_prefix_for_tag_at(idx)

            if token_tag == self.outside_tag:
                if self.is_span_under_construction():
                    self.close_span_under_construction_and_save_span(idx)
                else: 
                    continue
            else:
                if token_tag_prefix == self.start_prefix:
                    if self.is_span_under_construction():
                        self.close_span_under_construction_and_save_span(idx)

                    self.initiate_construction_of_new_span(idx)
                        
                elif token_tag_prefix == self.inside_prefix:
                    if self.is_span_under_construction():
                        continue
                    else:
                        self.initiate_construction_of_new_span(idx)

                elif token_tag_prefix not in self.valid_token_tag_prefix:
                    raise Exception(f'Unknown Token Tag: "{token_tag}" with prefix label: "{token_tag_prefix}" at idx: {idx}.')

        if self.is_span_under_construction():
            self.close_span_under_construction_and_save_span(len(self.tags))

        return self.spans


    def initiate_construction_of_new_span(self, start_idx) -> None:
        # temporarily create a span with same start and end idx, end idx can be modified later
        self.open_span_under_construction =  Span(self.get_label_for_tag_at(start_idx), 
                                                    start_idx, 
                                                    start_idx)

    def close_span_under_construction_and_save_span(self, stop_idx) -> None:
        start_idx =  self.open_span_under_construction.start_idx
        
        if not self.are_tokens_tagged_with_same_label(start_idx, stop_idx):
            raise Exception(f'Tokens with different labels found within the same span for idx range {(start_idx, stop_idx)}.')
        
        self.open_span_under_construction.end_idx = stop_idx-1
        self.open_span_under_construction.spanned_tokens = self.get_tokens(start_idx,stop_idx)
        self.open_span_under_construction.span_context =  self.get_tokens_with_context_tokens(start_idx, stop_idx)

        self.spans.append(self.open_span_under_construction)

        self.open_span_under_construction = None

    def are_tokens_tagged_with_same_label(self, start_idx, stop_idx):
        label_set = set()
        for idx in range(start_idx, stop_idx):
            label = self.get_label_for_tag_at(idx)
            label_set.add(label)
        return len(label_set)==1 or (start_idx==stop_idx and len(label_set)==0)

    def get_tokens(self, start_idx, stop_idx):
        return self.tokens[start_idx:stop_idx]

    def get_tokens_with_context_tokens(self, start_idx, stop_idx) -> List[str]: 
        context_start_idx = max(0, start_idx-self.context_padding)
        context_stop_idx = min(stop_idx+self.context_padding, len(self.tokens))
        return self.tokens[context_start_idx:context_stop_idx]

    def reset_spans_aggregator(self):
        self.spans = []

    def get_prefix_for_tag_at(self, idx):
        if self.tags[idx] == self.outside_tag:
            return self.outside_tag
        elif len(self.tags[idx])<=2:
            raise Exception(f'Tag length too short cannot extract a prefix from the tag "{self.tags[idx]}" at idx: {idx}.')
        else:
            return self.tags[idx][:1]

    def get_label_for_tag_at(self, idx):
        if self.tags[idx] == self.outside_tag:
            raise Exception(f'Cannot retrieve label for a token tagged with "{self.outside_tag}" at idx: {idx}.')
        elif len(self.tags[idx])<=2:
            raise Exception(f'Tag length too short cannot extract a label from the tag "{self.tags[idx]}" at idx: {idx}.')
        else:
            return self.tags[idx][2:]

    def is_span_under_construction(self):
        return self.open_span_under_construction is not None
