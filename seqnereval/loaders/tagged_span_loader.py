from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod
from ..models import Span

class TaggedSpanLoader(ABC):
    def __init__(self, tokens: List[str], tags: List[str], context_padding:int = 2):
        self.tokens = tokens
        self.tags = tags
        self.context_padding = context_padding

    @abstractmethod
    def retreive_spans(self)->List[Span]:
        pass
