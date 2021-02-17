# **seqnereval**: NER Evaluator
![Build - Main](https://github.com/ArshSekhon/pubtator_loader/workflows/Build%20-%20Main/badge.svg) [![codecov](https://codecov.io/gh/ArshSekhon/seqnereval/branch/main/graph/badge.svg?token=WTI9TUQ7E7)](https://codecov.io/gh/ArshSekhon/seqnereval) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

seqnereval is a Python module that allows you to perform extensive error analysis on your NER models. It is inspired by [nerevaluate](https://github.com/ivyleavedtoadflax/nervaluate) and is designed to be faster, easier to understand/extend and provide more granular insights on the errors made by the model. It allows you to:
- Check what was the type of errors were made by the model
- Find the exact entities that were misclassified or missed
- Get the context of these errors .

It draws heavily on [Segura-bedmar, I., & Mart, P. (2013). 2013 SemEval-2013 Task 9 Extraction of Drug-Drug Interactions from. Semeval](https://www.aclweb.org/anthology/S13-2056), 2(DDIExtraction), 341–350.

