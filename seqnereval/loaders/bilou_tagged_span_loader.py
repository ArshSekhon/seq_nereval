from __future__ import annotations
from typing import List
from . import BIOESTaggedSpanLoader

class BILOUTaggedSpanLoader(BIOESTaggedSpanLoader):
    def __init__(self, tokens: List[str], tags: List[str], context_padding:int = 2):
        super().__init__(tokens, tags, context_padding)

        self.start_prefix: str = 'B'
        self.inside_prefix: str = 'I'
        self.outside_tag: str = 'O'
        self.end_prefix: str = 'L'
        self.single_prefix: str = 'U'

        self.valid_token_tag_prefix = set([self.start_prefix, 
                                            self.inside_prefix, 
                                            self.outside_tag, 
                                            self.end_prefix, 
                                            self.single_prefix])
