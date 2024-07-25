# BM25CorpusSolver Plugin

BM25CorpusSolver is an OVOS (OpenVoiceOS) plugin designed to retrieve answers from a corpus of documents using the [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
algorithm. This solver is ideal for question-answering systems that require efficient and accurate retrieval of
information from a predefined set of documents.

- [Features](#features)
- [Retrieval Chatbots](#retrieval-chatbots)
   - [Example Solvers](#example-solvers)
     - [SquadQASolver](#squadqasolver)
     - [FreebaseQASolver](#freebaseqasolver)
  - [Implementing a Retrieval Chatbot](#implementing-a-retrieval-chatbot)
     - [BM25CorpusSolver](#bm25corpussolver)
     - [BM25QACorpusSolver](#bm25qacorpussolver)
  - [Limitations of Retrieval Chatbots](#limitations-of-retrieval-chatbots)
- [ReRanking](#reranking)
   - [BM25MultipleChoiceSolver](#bm25multiplechoicesolver)
   - [BM25EvidenceSolverPlugin](#bm25evidencesolverplugin)
- [Embeddings Store](#embeddings-store)
- [Integrating with Persona Framework](#integrating-with-persona-framework)


## Features

- **BM25 Algorithm**: Utilizes the BM25 ranking function for information retrieval, providing relevance-based document scoring.
- **Configurable**: Allows customization of language, minimum confidence score, and the number of answers to retrieve.
- **Logging**: Integrates with OVOS logging system for debugging and monitoring.
- **BM25QACorpusSolver**: Extends `BM25CorpusSolver` to handle question-answer pairs, optimizing the retrieval process for QA datasets.
- **BM25MultipleChoiceSolver**: Reranks multiple-choice options based on relevance to the query.
- **BM25EvidenceSolverPlugin**: Extracts the best sentence from a text that answers a question using the BM25 algorithm.

## Retrieval Chatbots

Retrieval chatbots use BM25CorpusSolver to provide answers to user queries by searching through a preloaded corpus of documents or QA pairs. 

These chatbots excel in environments where the information is structured and the queries are straightforward.

### Example solvers

#### SquadQASolver

The SquadQASolver is a subclass of BM25QACorpusSolver that automatically loads and indexes the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/) upon initialization.

This solver is suitable for usage with the ovos-persona framework.

```python
from ovos_bm25_solver import SquadQASolver

s = SquadQASolver()
query = "is there life on mars"
print("Query:", query)
print("Answer:", s.spoken_answer(query))
# 2024-07-19 22:31:12.625 - OVOS - __main__:load_corpus:60 - DEBUG - indexed 86769 documents
# 2024-07-19 22:31:12.625 - OVOS - __main__:load_squad_corpus:119 - INFO - Loaded and indexed 86769 question-answer pairs from SQuAD dataset
# Query: is there life on mars
# 2024-07-19 22:31:12.628 - OVOS - __main__:retrieve_from_corpus:69 - DEBUG - Rank 1 (score: 6.334013938903809): How is it postulated that Mars life might have evolved?
# 2024-07-19 22:31:12.628 - OVOS - __main__:retrieve_from_corpus:93 - DEBUG - closest question in corpus: How is it postulated that Mars life might have evolved?
# Answer: similar to Antarctic
```

#### FreebaseQASolver

The FreebaseQASolver is a subclass of BM25QACorpusSolver that automatically loads and indexes the [FreebaseQA dataset](https://github.com/kelvin-jiang/FreebaseQA) upon initialization.

This solver is suitable for usage with the ovos-persona framework.

```python
from ovos_bm25_solver import FreebaseQASolver

s = FreebaseQASolver()
query = "What is the capital of France"
print("Query:", query)
print("Answer:", s.spoken_answer(query))
# 2024-07-19 22:31:09.468 - OVOS - __main__:load_corpus:60 - DEBUG - indexed 20357 documents
# Query: What is the capital of France
# 2024-07-19 22:31:09.468 - OVOS - __main__:retrieve_from_corpus:69 - DEBUG - Rank 1 (score: 5.996074199676514): what is the capital of france
# 2024-07-19 22:31:09.469 - OVOS - __main__:retrieve_from_corpus:93 - DEBUG - closest question in corpus: what is the capital of france
# Answer: paris
```

### Implementing a Retrieval Chatbot

To use the BM25CorpusSolver, you need to create an instance of the solver, load your corpus, and then query it.

#### BM25CorpusSolver

This class is meant to be used to create your own solvers with a dedicated corpus.

```python
from ovos_bm25_solver import BM25CorpusSolver

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

#### BM25QACorpusSolver

This class is meant to be used to create your own solvers with a dedicated corpus

BM25QACorpusSolver is an extension of BM25CorpusSolver, designed to work with question-answer pairs. It is particularly
useful when working with datasets like SQuAD, FreebaseQA, or similar QA datasets.

```python
import requests
from ovos_bm25_solver import BM25QACorpusSolver

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

In this example, BM25QACorpusSolver is used to load a large corpus of question-answer pairs from the SQuAD and
FreebaseQA datasets. The solver retrieves the best matching answer for the given query.

### Limitations of Retrieval Chatbots

Retrieval chatbots, while powerful, have certain limitations. These include:

1. **Dependence on Corpus Quality and Size**: The accuracy of a retrieval chatbot heavily relies on the quality and comprehensiveness of the underlying corpus. A limited or biased corpus can lead to inaccurate or irrelevant responses.
2. **Static Knowledge Base**: Unlike generative models, retrieval chatbots can't generate new information or answers. They can only retrieve and rephrase content from the pre-existing corpus.
3. **Contextual Understanding**: While advanced algorithms like BM25 can rank documents based on relevance, they may still struggle with understanding nuanced or complex queries, especially those requiring deep contextual understanding.
4. **Scalability**: As the size of the corpus increases, the computational resources required for indexing and retrieving relevant documents also increase, potentially impacting performance.
5. **Dynamic Updates**: Keeping the corpus updated with the latest information can be challenging, especially in fast-evolving domains.

Despite these limitations, retrieval chatbots are effective for domains where the corpus is well-defined and relatively static, such as FAQs, documentation, and knowledge bases.

### ReRanking

ReRanking is a technique used to refine a list of potential answers by evaluating their relevance to a given query. 
This process is crucial in scenarios where multiple options or responses need to be assessed to determine the most appropriate one.

In retrieval chatbots, ReRanking helps in selecting the best answer from a set of retrieved documents or options, enhancing the accuracy of the response provided to the user.

`MultipleChoiceSolver` are integrated into the OVOS Common Query framework, where they are used to select the most relevant answer from a set of multiple skill responses.

#### BM25MultipleChoiceSolver

BM25MultipleChoiceSolver is designed to select the best answer to a question from a list of options.

In the context of retrieval chatbots, BM25MultipleChoiceSolver is useful for scenarios where a user query results in a list of predefined answers or options. 
The solver ranks these options based on their relevance to the query and selects the most suitable one.


```python
from ovos_bm25_solver import BM25MultipleChoiceSolver

solver = BM25MultipleChoiceSolver()
a = solver.rerank("what is the speed of light", [
    "very fast", "10m/s", "the speed of light is C"
])
print(a)
# 2024-07-22 15:03:10.295 - OVOS - __main__:load_corpus:61 - DEBUG - indexed 3 documents
# 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 1 (score: 0.7198746800422668): the speed of light is C
# 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 2 (score: 0.0): 10m/s
# 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 3 (score: 0.0): very fast
# [(0.7198747, 'the speed of light is C'), (0.0, '10m/s'), (0.0, 'very fast')]

# NOTE: select_answer is part of the MultipleChoiceSolver base class and uses rerank internally
a = solver.select_answer("what is the speed of light", [
    "very fast", "10m/s", "the speed of light is C"
])
print(a) # the speed of light is C
```

#### BM25EvidenceSolverPlugin

BM25EvidenceSolverPlugin is designed to extract the most relevant sentence from a text passage that answers a given question. This plugin uses the BM25 algorithm to evaluate and rank sentences based on their relevance to the query.

In text extraction and machine comprehension tasks, BM25EvidenceSolverPlugin enables the identification of specific sentences within a larger body of text that directly address a user's query. 

For example, in a scenario where a user queries about the number of rovers exploring Mars, BM25EvidenceSolverPlugin scans the provided text passage, ranks sentences based on their relevance, and extracts the most informative sentence.

```python
from ovos_bm25_solver import BM25EvidenceSolverPlugin

config = {
    "lang": "en-us",
    "min_conf": 0.4,
    "n_answer": 1
}
solver = BM25EvidenceSolverPlugin(config)

text = """Mars is the fourth planet from the Sun. It is a dusty, cold, desert world with a very thin atmosphere. 
Mars is also a dynamic planet with seasons, polar ice caps, canyons, extinct volcanoes, and evidence that it was even more active in the past.
Mars is one of the most explored bodies in our solar system, and it's the only planet where we've sent rovers to roam the alien landscape. 
NASA currently has two rovers (Curiosity and Perseverance), one lander (InSight), and one helicopter (Ingenuity) exploring the surface of Mars.
"""
query = "how many rovers are currently exploring Mars"
answer = solver.get_best_passage(evidence=text, question=query)
print("Query:", query)
print("Answer:", answer)
# 2024-07-22 15:05:14.209 - OVOS - __main__:load_corpus:61 - DEBUG - indexed 5 documents
# 2024-07-22 15:05:14.209 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 1 (score: 1.39238703250885): NASA currently has two rovers (Curiosity and Perseverance), one lander (InSight), and one helicopter (Ingenuity) exploring the surface of Mars.
# 2024-07-22 15:05:14.210 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 2 (score: 0.38667747378349304): Mars is one of the most explored bodies in our solar system, and it's the only planet where we've sent rovers to roam the alien landscape.
# 2024-07-22 15:05:14.210 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 3 (score: 0.15732118487358093): Mars is the fourth planet from the Sun.
# 2024-07-22 15:05:14.210 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 4 (score: 0.10177625715732574): Mars is also a dynamic planet with seasons, polar ice caps, canyons, extinct volcanoes, and evidence that it was even more active in the past.
# 2024-07-22 15:05:14.210 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 5 (score: 0.0): It is a dusty, cold, desert world with a very thin atmosphere.
# Query: how many rovers are currently exploring Mars
# Answer: NASA currently has two rovers (Curiosity and Perseverance), one lander (InSight), and one helicopter (Ingenuity) exploring the surface of Mars.

```

In this example, `BM25EvidenceSolverPlugin` effectively identifies and retrieves the most relevant sentence from the provided text that answers the query about the number of rovers exploring Mars. 
This capability is essential for applications requiring information extraction from extensive textual content, such as automated research assistants or content summarizers.

## Embeddings Store

A fake embeddings store is provided using only text search

> NOTE: this does not scale to large datasets

```python
from ovos_bm25_solver.embed import JsonEmbeddingsDB, BM25TextEmbeddingsStore
db = JsonEmbeddingsDB("bm25_index")
# Initialize the BM25 text embeddings store
index = BM25TextEmbeddingsStore(db=db)

# Add documents to the database
text = "hello world"
text2 = "goodbye cruel world"
index.add_document(text)
index.add_document(text2)

# Querying with fuzzy match
results = db.query("the world", top_k=2)
print("Fuzzy Match Results:", results)

# Querying with BM25
results = index.query("the world", top_k=2)
print("BM25 Query Results:", results)

# Comparing strings using fuzzy match
distance = index.distance(text, text2)
print("Distance between strings:", distance)
```

## Integrating with Persona Framework

To use the `SquadQASolver` and `FreebaseQASolver` in the persona framework, you can define a persona configuration file and specify the solvers to be used.

Here's an example of how to define a persona that uses the `SquadQASolver` and `FreebaseQASolver`:

1. Create a persona configuration file, e.g., `qa_persona.json`:

```json
{
  "name": "QAPersona",
  "solvers": [
    "ovos-solver-squadqa-plugin",
    "ovos-solver-freebaseqa-plugin",
    "ovos-solver-failure-plugin"
  ]
}
```

2. Run [ovos-persona-server](https://github.com/OpenVoiceOS/ovos-persona-server) with the defined persona:

```bash
$ ovos-persona-server --persona qa_persona.json
```

In this example, the persona named "QAPersona" will first use the `SquadQASolver` to answer questions. If it cannot find an answer, it will fall back to the `FreebaseQASolver`. Finally, it will use the `ovos-solver-failure-plugin` to ensure it always responds with something, even if the previous solvers fail.


Check setup.py for reference in how to package your own corpus backed solvers

```python
PLUGIN_ENTRY_POINTS = [
    'ovos-solver-bm25-squad-plugin=ovos_bm25_solver:SquadQASolver',
    'ovos-solver-bm25-freebase-plugin=ovos_bm25_solver:FreebaseQASolver'
]
```
