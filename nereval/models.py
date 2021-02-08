from copy import deepcopy

class NERResult:
    def __init__(self):
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

class NERSpan:
    def __init__(self, entity_type:str, start_idx:int, end_idx:int):
        self.entity_type = entity_type
        self.start_idx = start_idx
        self.end_idx = end_idx