# BM25CorpusSolver Plugin

BM25CorpusSolver is an OVOS (OpenVoiceOS) plugin designed to retrieve answers from a corpus of documents using the BM25 algorithm. This solver is ideal for question-answering systems that require efficient and accurate retrieval of information from a predefined set of documents.

## Features

- **BM25 Algorithm**: Utilizes the BM25 ranking function for information retrieval, providing relevance-based document scoring.
- **Configurable**: Allows customization of language, minimum confidence score, and the number of answers to retrieve.
- **Logging**: Integrates with OVOS logging system for debugging and monitoring.
- **BM25QACorpusSolver**: Extends `BM25CorpusSolver` to handle question-answer pairs, optimizing the retrieval process for QA datasets.

## Installation

To install BM25CorpusSolver, ensure you have the necessary dependencies:

```bash
pip install ovos-plugin-manager ovos-utils bm25s
```

## Usage

To use the BM25CorpusSolver, you need to create an instance of the solver, load your corpus, and then query it.

You can customize the solver's configuration by passing a dictionary during initialization:

### BM25CorpusSolver Example

```python
config = {
    "lang": "en-us",
    "min_conf": 0.4,
    "n_answer": 2
}
solver = BM25CorpusSolver(config)

corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a fish is a creature that lives in water and swims",
]
solver.load_corpus(corpus)

query = "does the fish purr like a cat?"
answer = solver.get_spoken_answer(query)
print(answer)

# Expected Output:
# 2024-07-19 20:03:29.979 - OVOS - ovos_plugin_manager.utils.config:get_plugin_config:40 - DEBUG - Loaded configuration: {'module': 'ovos-translate-plugin-server', 'lang': 'en-us'}
# 2024-07-19 20:03:30.024 - OVOS - __main__:load_corpus:28 - DEBUG - indexed 4 documents
# 2024-07-19 20:03:30.025 - OVOS - __main__:retrieve_from_corpus:37 - DEBUG - Rank 1 (score: 1.0584375858306885): a cat is a feline and likes to purr
# 2024-07-19 20:03:30.025 - OVOS - __main__:retrieve_from_corpus:37 - DEBUG - Rank 2 (score: 0.481589138507843): a fish is a creature that lives in water and swims
# a cat is a feline and likes to purr. a fish is a creature that lives in water and swims
```

### BM25QACorpusSolver Example

BM25QACorpusSolver is an extension of BM25CorpusSolver, designed to work with question-answer pairs. It is particularly useful when working with datasets like SQuAD, FreebaseQA, or similar QA datasets.

```python
import requests

# Load SQuAD dataset
corpus = {}
data = requests.get("https://github.com/chrischute/squad/raw/master/data/train-v2.0.json").json()
for s in data["data"]:
    for p in s["paragraphs"]:
        for qa in p["qas"]:
            if "question" in qa and qa["answers"]:
                corpus[qa["question"]] = qa["answers"][0]["text"]

# Load FreebaseQA dataset
data = requests.get("https://github.com/kelvin-jiang/FreebaseQA/raw/master/FreebaseQA-train.json").json()
for qa in data["Questions"]:
    q = qa["ProcessedQuestion"]
    a = qa["Parses"][0]["Answers"][0]["AnswersName"][0]
    corpus[q] = a

# Initialize BM25QACorpusSolver with config
config = {
    "lang": "en-us",
    "min_conf": 0.4,
    "n_answer": 1
}
solver = BM25QACorpusSolver(config)
solver.load_corpus(corpus)

query = "is there life on mars?"
answer = solver.get_spoken_answer(query)
print("Query:", query)
print("Answer:", answer)

# Expected Output:
# 86769 qa pairs imports from squad dataset
# 20357 qa pairs imports from freebaseQA dataset
# 2024-07-19 21:49:31.360 - OVOS - ovos_plugin_manager.language:create:233 - INFO - Loaded the Language Translation plugin ovos-translate-plugin-server
# 2024-07-19 21:49:31.360 - OVOS - ovos_plugin_manager.utils.config:get_plugin_config:40 - DEBUG - Loaded configuration: {'module': 'ovos-translate-plugin-server', 'lang': 'en-us'}
# 2024-07-19 21:49:32.759 - OVOS - __main__:load_corpus:61 - DEBUG - indexed 107126 documents
# Query: is there life on mars
# 2024-07-19 21:49:32.760 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 1 (score: 6.037893295288086): How is it postulated that Mars life might have evolved?
# 2024-07-19 21:49:32.760 - OVOS - __main__:retrieve_from_corpus:94 - DEBUG - closest question in corpus: How is it postulated that Mars life might have evolved?
# Answer: similar to Antarctic
```

In this example, BM25QACorpusSolver is used to load a large corpus of question-answer pairs from the SQuAD and FreebaseQA datasets. The solver retrieves the best matching answer for the given query.

## SquadQASolver

The SquadQASolver is a subclass of BM25QACorpusSolver that automatically loads and indexes the SQuAD dataset upon initialization.

```python
solver = SquadQASolver()

query = "What is the capital of France?"
answer = solver.get_spoken_answer(query)
print(answer)

# Expected Output:
# Paris
```
