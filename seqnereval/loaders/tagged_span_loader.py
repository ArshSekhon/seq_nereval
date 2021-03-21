from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod
from ..models import Span

class TaggedSpanLoader(ABC):
    def __init__(self, tokens: List[str], tags: List[str]):
        self.tokens = tokens
        self.tags = tags

    @abstractmethod
    def retreive_spans(self)->List[Span]:
        pass
