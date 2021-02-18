# **seqnereval**: NER Model Evaluator
![Build - Main](https://github.com/ArshSekhon/pubtator_loader/workflows/Build%20-%20Main/badge.svg) [![codecov](https://codecov.io/gh/ArshSekhon/seqnereval/branch/main/graph/badge.svg?token=WTI9TUQ7E7)](https://codecov.io/gh/ArshSekhon/seqnereval) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![PyPI version](https://badge.fury.io/py/seqnereval.svg)](https://badge.fury.io/py/seqnereval)

`seqnereval` is a Python module that allows you to efficiently perform extensive error analysis on your NER models. It allows you to:
- Check what was the type of errors were made by the model.
- Find the exact entities that were misclassified or missed.
- Get the context of these errors.

One of the key motivation behind writing this module was to provide an easier and more optimal way of evaluating NER models. It was inspired by some existing NER model evaluation and was designed keeping performance in mind, so you can get your results faster than most of the existing NER evaluation packages.
## Installation
To install simply execute:
```sh
pip install seqnereval
```

## Usage
```py
from seqnereval import NERTagListEvaluator

# list of lists of tokens for different docs
tokens_lists = [
    ['The', 'John', 'Doe\'s', 'Basketball', 'Club'], # Doc 1
    ['The', 'Canada', 'Place', 'is', 'best', '.'], # Doc 2
    ['Other', 'John', 'is', 'a', 'good', 'person', '.'], # Doc 3
    ['John', 'Doe', 'Jenny', 'Doe', '_', '_'], # Doc 4
]

# list of lists of predicted tags for different docs
predicted_tag_lists = [
    ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"], # Doc 1
    ["O", "B-LOC", "I-LOC", "O", "O", "O"], # Doc 2
    ["O", "U-PER", "O", "O", "O", "O", "O"], # Doc 3
    ["B-PER", "I-PER", "B-PER", "I-PER", "O", "O"], # Doc 4
]

# list of lists of golden/true tags for different docs
gold_tag_lists = [
    ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"], # Doc 1
    ["O", "B-LOC", "I-LOC", "O", "O", "O"], # Doc 2
    ["O", "U-PER", "O", "O", "O", "O", "O"], # Doc 3
    ["B-PER", "I-PER", "B-PER", "I-PER", "O", "O"], # Doc 4
]

    
evaluator = NERTagListEvaluator(tokens_lists, gold_tag_lists, predicted_tag_lists, 2)
result, results_by_tags = evaluator.evaluate()
# Refer to the next section (Understanding the results) to find how to use
# result object obtained to get more information.

# For e.g. results can be summarized as follows
print(result.summarize_result())
"""
OUTPUT: 

{'strict_match': {'correct': 34926,
  'incorrect': 23323,
  'partial': 0,
  'missed': 7319,
  'spurious': 6002,
  'possible': 65568,
  'actual': 64251,
  'precision': 0.5435868702432647,
  'recall': 0.5326683748169839,
  'f1': 0.5380722390405103},
 'type_match': {'correct': 42283,
  'incorrect': 15966,
  'partial': 0,
    .
    .
    .
 'partial_match': {'correct': 41668,
    .
    .
    .
 'bounds_match': {'correct': 41668,
    .
    .
    .
}
"""
```


## Extracting and Undestanding the Results
`seqnereval` identifies the error made by an NER model while tagging the entities in a sequence and classifies these errors into following 6 categories:


__Type 1. Entity Type and Span match__

|Token|Gold|Prediction|
|---|---|---|
|Vancouver|B-LOC|B-LOC|
|Island|I-LOC|I-LOC|
|is|O|O|
|the|O|O|

__Type 2. Predicted Entity is not an entity according to golden dataset__

|Token|Gold|Prediction|
|---|---|---|
|is|O|O|
|an|O|B-PER|
|extremely|O|I-PER|
|desireable|O|O|

__Type 3. Entity is not predicted by the system__

|Token|Gold|Prediction|
|---|---|---|
|Vancouver|B-LOC|O|
|Island|I-LOC|O|
|is|O|O|
|the|O|O|

__Type 4. Entity type is wrong but the span is correct__

|Token|Gold|Prediction|
|---|---|---|
|I|O|O|
|live|O|O|
|in|O|O|
|Palo|B-LOC|B-ORG|
|Alto|I-LOC|I-ORG|
|,|O|O|

__Type 5. System gets the boundaries of the surface string wrong__

|Token|Gold|Prediction|
|---|---|---|
|Unless|O|B-PER|
|Karl|B-PER|I-PER|
|Smith|I-PER|I-PER|
|resigns|O|O|

__Type 6. System gets the boundaries and entity type wrong__

|Token|Gold|Prediction|
|---|---|---|
|Unless|O|B-ORG|
|Karl|B-PER|I-ORG|
|Smith|I-PER|I-OR

Predicted Entities and their corresponding Gold/True entities (if applicable) that fall into each of these categories can be obtained as follows:

```py
...
...
evaluator =  NERTagListEvaluator(
                     # list of lists of tokens, 
                     # e.g. [[tokens for doc 1..],[tokens for doc 2..]...]
                    list_of_token_lists, 
                    # list of lists of gold tags, 
                    # e.g. [[gold tags for doc 1..],[gold tags for doc 2..]...]
                    list_of_gold_tag_lists, 
                    # list of lists of predicted tags, 
                    # e.g. [[predicted tags for doc 1..],[predicted tags for doc 2..]...
                    list_of_predicted_tag_lists
                )
results, results_by_tags = evaluator.evaluate()

print(results.type_match_span_match)

"""
OUTPUT:
  
[
    {Gold: (Entity Type: "T103", Token Span IDX:(0, 1), Tokens:['Nonylphenol', 'diethoxylate'], Context:['Nonylphenol', 'diethoxylate', 'inhibits', 'apoptosis']), 
    Predicted: (Entity Type: "T103", Token Span IDX:(0, 1), Tokens:['Nonylphenol', 'diethoxylate'], Context:['Nonylphenol', 'diethoxylate', 'inhibits', 'apoptosis'])}, 
    
    {Gold: (Entity Type: "T038", Token Span IDX:(3, 3), Tokens:['apoptosis'], Context:['diethoxylate', 'inhibits', 'apoptosis', 'induced', 'in']), 
    Predicted: (Entity Type: "T038", Token Span IDX:(3, 3), Tokens:['apoptosis'], Context:['diethoxylate', 'inhibits', 'apoptosis', 'induced', 'in'])}, 
    
    {Gold: (Entity Type: "T169", Token Span IDX:(4, 4), Tokens:['induced'], Context:['inhibits', 'apoptosis', 'induced', 'in', 'PC12']), 
    Predicted: (Entity Type: "T169", Token Span IDX:(4, 4), Tokens:['induced'], Context:['inhibits', 'apoptosis', 'induced', 'in', 'PC12'])}
    .
    .
    .
]
"""

# similarily the entities in other categories can be accessed in the similar way

print(results.unecessary_predicted_entity) # Type 2
print(results.missed_gold_entity) # Type 3
print(results.type_mismatch_span_match) # Type 4
print(results.type_match_span_partial) # Type 5
print(results.type_mismatch_span_partial) # Type 6
```

Following five metrics are used to consider difference categories of errors:

|Error type|Explanation|
|---|---|
|Correct (COR)|both are the same|
|Incorrect (INC)|the output of a system and the golden annotation don’t match|
|Partial (PAR)|system and the golden annotation are somewhat “similar” but not the same|
|Missing (MIS)|a golden annotation is not captured by a system|
|Spurius (SPU)|system produces a response which doesn’t exit in the golden annotation|

These metrics are measured in following four different ways:

|Evaluation schema|Explanation|
|---|---|
|Strict Match|exact boundary surface string match and entity type|
|Bount Match|exact boundary match over the surface string, regardless of the type|
|Partial Match|partial boundary match over the surface string, regardless of the type|
|Type Match|some overlap between the system tagged entity and the gold annotation is required|

These five errors and four evaluation schema interact in the following ways:

|Scenario|Gold entity|Gold string|Pred entity|Pred string|Type Match|Partial Match|Bound Match|Strict Match|
|---|---|---|---|---|---|---|---|---|
|I|PER|John|PER|John|COR|COR|COR|COR|
|II| | |LOC|extreme|SPU|SPU|SPU|SPU|
|III|LOC|Germany| | |MIS|MIS|MIS|MIS|
|IV|LOC|vancouver island|ORG|vancouver island|INC|COR|COR|INC|
|V|LOC|Detroit|LOC|in Detroit|COR|PAR|INC|INC|
|VI|LOC|Detroit|ORG|in Detroit|INC|PAR|INC|INC|

The entity spans falling into each of these categories can be obtained as follows:
```py
...
...
evaluator =  NERTagListEvaluator(
                     # list of lists of tokens, 
                     # e.g. [[tokens for doc 1..],[tokens for doc 2..]...]
                    list_of_token_lists, 
                    # list of lists of gold tags, 
                    # e.g. [[gold tags for doc 1..],[gold tags for doc 2..]...]
                    list_of_gold_tag_lists, 
                    # list of lists of predicted tags, 
                    # e.g. [[predicted tags for doc 1..],[predicted tags for doc 2..]...
                    list_of_predicted_tag_lists
                )

results, results_by_tags = evaluator.evaluate()

# Strict Match
print(results.strict_match["correct"])
print(results.strict_match["incorrect"])
print(results.strict_match["missed"])
print(results.strict_match["spurious"])

print(results.strict_match["precision"])
print(results.strict_match["recall"])
print(results.strict_match["f1"])

# Type Match
print(results.type_match["correct"])
print(results.type_match["incorrect"])
print(results.type_match["missed"])
print(results.type_match["spurious"])

print(results.type_match["precision"])
print(results.type_match["recall"])
print(results.type_match["f1"])

# Partial Match
print(results.partial_match["correct"])
print(results.partial_match["incorrect"])
print(results.partial_match["missed"])
print(results.partial_match["spurious"])

print(results.partial_match["precision"])
print(results.partial_match["recall"])
print(results.partial_match["f1"])

# Bounds/Exact Match
print(results.bounds_match["correct"])
print(results.bounds_match["incorrect"])
print(results.bounds_match["missed"])
print(results.bounds_match["spurious"])

print(results.bounds_match["precision"])
print(results.bounds_match["recall"])
print(results.bounds_match["f1"])

```



Precision/Recall/F1-score are calculated for each different evaluation schema as follows:

__For Strict Match and Bounds Match__
```
Precision = (COR / ACT) = TP / (TP + FP)
Recall = (COR / POS) = TP / (TP+FN)
```
__For Partial Match and Type Match__
```
Precision = (COR + 0.5 × PAR) / ACT = TP / (TP + FP)
Recall = (COR + 0.5 × PAR)/POS = COR / ACT = TP / (TP + FP)
```
where:

```
POSSIBLE (POS) = COR + INC + PAR + MIS = TP + FN
ACTUAL (ACT) = COR + INC + PAR + SPU = TP + FP
```




## References
`seqnereval` draws heavily on [Segura-bedmar, I., & Mart, P. (2013). 2013 SemEval-2013 Task 9 Extraction of Drug-Drug Interactions from. Semeval](https://www.aclweb.org/anthology/S13-2056), 2(DDIExtraction), 341–350.  It was inspired by [nerevaluate](https://github.com/ivyleavedtoadflax/nervaluate) and is designed to be significantly faster, easier to understand/extend and provide more granular insights on the nature of errors made by the model.
