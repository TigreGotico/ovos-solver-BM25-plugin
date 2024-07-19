from typing import List, Iterable, Optional
from ovos_plugin_manager.templates.solvers import MultipleChoiceSolver, QuestionSolver
from ovos_utils.log import LOG

import bm25s


class BM25CorpusSolver(QuestionSolver):
    enable_tx = False
    priority = 60

    def __init__(self, config=None):
        config = config or {"lang": "en-us",
                            "min_conf": 0.4,
                            "n_answer": 2}
        super().__init__(config)
        # Create the BM25 model
        self.retriever = bm25s.BM25()
        self.corpus = None

    def load_corpus(self, corpus: List[str]):
        self.corpus = corpus
        # Tokenize the corpus and only keep the ids (faster and saves memory)
        corpus_tokens = bm25s.tokenize(corpus, stopwords=self.default_lang.split("-")[0])
        # index the corpus
        self.retriever.index(corpus_tokens)
        LOG.debug(f"indexed {len(corpus)} documents")

    def retrieve_from_corpus(self,query,  k=3) -> str:
        # Query the corpus
        query_tokens = bm25s.tokenize(query, stopwords=self.default_lang.split("-")[0])
        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        results, scores = self.retriever.retrieve(query_tokens, corpus=corpus, k=k)
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            LOG.debug(f"Rank {i + 1} (score: {score}): {doc}")
            if score >= self.config.get("min_conf", 0.4):
                yield doc

    def get_spoken_answer(self, query: str, context: Optional[dict] = None) -> str:
        if self.corpus is None:
            return None
        # Query the corpus
        answers = list(self.retrieve_from_corpus(query, k=self.config.get("n_answer", 1)))
        if answers:
            return ". ".join(answers)


if __name__ == "__main__":
    LOG.set_level("DEBUG")
    # Create your corpus here
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ]

    s = BM25CorpusSolver()
    s.load_corpus(corpus)

    query = "does the fish purr like a cat?"
    print(s.spoken_answer(query))

    # 2024-07-19 20:03:29.979 - OVOS - ovos_plugin_manager.utils.config:get_plugin_config:40 - DEBUG - Loaded configuration: {'module': 'ovos-translate-plugin-server', 'lang': 'en-us'}
    # 2024-07-19 20:03:30.024 - OVOS - __main__:load_corpus:28 - DEBUG - indexed 4 documents
    # 2024-07-19 20:03:30.025 - OVOS - __main__:retrieve_from_corpus:37 - DEBUG - Rank 1 (score: 1.0584375858306885): a cat is a feline and likes to purr
    # 2024-07-19 20:03:30.025 - OVOS - __main__:retrieve_from_corpus:37 - DEBUG - Rank 2 (score: 0.481589138507843): a fish is a creature that lives in water and swims
    # a cat is a feline and likes to purr. a fish is a creature that lives in water and swims
