from copy import deepcopy


class NERResult:
    def __init__(self):
        """
        Constructor for NERResult 
        """
        self.__metrics_template = {
            "correct": [],
            "incorrect": [],
            "partial": [],
            "missed": [],
            "spurious": [],
            "possible": [],
            "actual": [],
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

        self.strict_match = deepcopy(self.__metrics_template)
        self.type_match = deepcopy(self.__metrics_template)
        self.partial_match = deepcopy(self.__metrics_template)
        self.bounds_match = deepcopy(self.__metrics_template)


class NEREntitySpan:
    def __init__(self, entity_type: str, start_idx: int, end_idx: int):
        """
        Constructor for NEREntitySpan

        Parameters:
            entity_type (str): type/label of entity.
            start_idx (int): index of the first token that is a part of the entity.
            end_idx (int): index of the last token that is a part of the entity.
        """
        self.entity_type = entity_type
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __hash__(self):
        return hash(f'{self.entity_type}-{self.start_idx}-{self.end_idx}')

    def __eq__(self, other):
        return (self.entity_type == other.entity_type and
                self.start_idx == other.start_idx and
                self.end_idx == other.end_idx)

    def spans_same_tokens_as(self, otherEntity: NEREntitySpan):
        """
        Finds if both the entities span same tokens regardless of type.

        Parameters:
            otherEntity (NEREntitySpan): other entity to check
        Returns:
            'True' if they both span the same tokens, else 'False'
        """
        return (self.start_idx == otherEntity.start_idx and self.end_idx == otherEntity.end_idx)

    def overlaps_with(self, otherEntity: NEREntitySpan) -> bool:
        """
        Finds if the given span overlaps

        Parameters:
            otherEntity (NEREntitySpan): other span to check for overlap
        Returns:
            'True' if there is an overlap, else 'False'
        """
        return max(self.start_idx, otherEntity.start_idx) <= min(self.end_idx, otherEntity.end_idx)
