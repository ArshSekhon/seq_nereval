from .models import NERResult, NERSpan
from typing import List

class NEREvaluator:
    def __init__(self, gold_entity_span_lists: List[List[NERSpan]], pred_entity_span_lists: List[List[NERSpan]]):
        """
        Constructor for NEREvaluator

        Parameters:
            gold_entity_span_lists (List[List[NERSpan]]): List of lists of spans for the gold entities.
            pred_entity_span_lists (List[List[NERSpan]]): List of lists of spans for the predicted entities.
        """
        self.gold_entity_span_lists = gold_entity_span_lists
        self.pred_entity_span_lists = pred_entity_span_lists
        self.unique_gold_tags = list(
            set([span.entity_type 
                    for gold_entity_span_list in gold_entity_span_lists 
                        for span in gold_entity_span_list]))

        self.results = NERResult()
        self.results_grouped_by_tags = {
            tag: NERResult() for tag in self.unique_gold_tags
        }




class NERTagListEvaluator(NEREvaluator):
    def __init__(self, tokens: List[List[str]], gold_tag_lists: List[List[str]], pred_tag_lists: List[List[str]]):
        """
            Constructor for tag list based evaluator

            Parameters:
                tokens (List[List[str]]): List of lists of tokens.
                gold_tag_lists (List[List[str]]): List of lists of golden tags.
                pred_tag_lists (List[List[str]]): List of lists of predicted tags.
        """

        self.tokens = tokens
        self.gold_tag_lists = gold_tag_lists
        self.pred_tag_lists = pred_tag_lists

        gold_entity_spans = self.__tagged_list_to_span(
            self.gold_tag_lists)
        pred_entity_spans = self.__tagged_list_to_span(
            self.pred_tag_lists)

        super().__init__(gold_entity_spans, pred_entity_spans)

    def __tagged_list_to_span(self, tag_lists: List[List[str]]):
        """
            Create a list of tagged entities with span offsets.

            Parameters:
                tag_list (List[List[str]]): List of lists of tags
            Returns:
                List of different tagged entities with span offsets.
        """
        results = []
        start_offset = None
        end_offset = None
        label = None

        for tag_list in tag_lists:
            labelled_entities = []
            for offset, token_tag in enumerate(tag_list):

                if token_tag == "O":
                    # if a sequence of non-"O" tags was seen last and
                    # a "O" tag is encountered => Label has ended.
                    if label is not None and start_offset is not None:
                        end_offset = offset - 1
                        labelled_entities.append(
                            NERSpan(label, start_offset, end_offset)
                        )
                        start_offset = None
                        end_offset = None
                        label = None
                # if a non-"O" tag is encoutered => new label has started
                elif label is None:
                    label = token_tag[2:]
                    start_offset = offset
                # if another label begins => last labelled seq has ended
                elif label != token_tag[2:] or (
                    label == token_tag[2:] and token_tag[:1] == "B"
                ):

                    end_offset = offset - 1
                    labelled_entities.append(
                        NERSpan(label, start_offset, end_offset)
                    )

                    # start of a new label
                    label = token_tag[2:]
                    start_offset = offset
                    end_offset = None

            if label is not None and start_offset is not None and end_offset is None:
                labelled_entities.append(
                    NERSpan(label, start_offset, len(tag_list) - 1)
                )
            if len(labelled_entities) > 0:
                results.append(labelled_entities)
                labelled_entities = []

        return results
