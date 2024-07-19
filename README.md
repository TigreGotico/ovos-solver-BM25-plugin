# BM25CorpusSolver Plugin

BM25CorpusSolver is an OVOS (OpenVoiceOS) plugin designed to retrieve answers from a corpus of documents using the BM25 algorithm. This solver is ideal for question-answering systems that require efficient and accurate retrieval of information from a predefined set of documents.

## Features

- **BM25 Algorithm**: Utilizes the BM25 ranking function for information retrieval, providing relevance-based document scoring.
- **Configurable**: Allows customization of language, minimum confidence score, and the number of answers to retrieve.
- **Logging**: Integrates with OVOS logging system for debugging and monitoring.

## Installation

To install BM25CorpusSolver, ensure you have the necessary dependencies:

```bash
pip install ovos-plugin-manager ovos-utils bm25s
```

## Usage

To use the BM25CorpusSolver, you need to create an instance of the solver, load your corpus, and then query it.

You can customize the solver's configuration by passing a dictionary during initialization:

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
