from typing import List, Optional, Dict

import bm25s
import requests
from ovos_plugin_manager.templates.solvers import QuestionSolver
from ovos_utils.log import LOG


class BM25CorpusSolver(QuestionSolver):
    enable_tx = False
    priority = 60
    METHODS = ["robertson", "lucene", "bm25l", "bm25+", "atire", "rank-bm25", "bm25-pt"]
    IDF_METHODS = ['robertson', 'lucene', 'atire', 'bm25l', 'bm25+']

    def __init__(self, config=None):
        config = config or {"lang": "en-us",
                            "min_conf": 0.4,
                            "n_answer": 1,
                            "method": None,
                            "idf_method": None}
        super().__init__(config)
        # Create the BM25 model
        self.retriever = None
        self.corpus = None

    @property
    def method(self):
        m = self.config.get("method")
        if m is None:
            return None
        if m not in self.METHODS:
            LOG.warning(f"{m} is not a valid method, choose one of {self.METHODS}")
            m = None
        return m

    @property
    def idf_method(self):
        m = self.config.get("idf_method")
        if m is None:
            return None
        if m not in self.IDF_METHODS:
            LOG.warning(f"{m} is not a valid method, choose one of {self.IDF_METHODS}")
            m = "lucene"
        return m

    def load_corpus(self, corpus: List[str]):
        if self.method == "rank-bm25":
            self.retriever = bm25s.BM25(method="atire", idf_method="robertson")
        elif self.method == "bm25-pt":
            self.retriever = bm25s.BM25(method="atire", idf_method="lucene")
        elif self.method is not None:
            self.retriever = bm25s.BM25(method=self.method, idf_method=self.idf_method)
        else:
            self.retriever = bm25s.BM25()
        self.corpus = corpus
        # Tokenize the corpus and only keep the ids (faster and saves memory)
        corpus_tokens = bm25s.tokenize(corpus, stopwords=self.default_lang.split("-")[0])
        # index the corpus
        self.retriever.index(corpus_tokens)
        LOG.debug(f"indexed {len(corpus)} documents")

    def retrieve_from_corpus(self, query, k=3) -> str:
        # Query the corpus
        query_tokens = bm25s.tokenize(query, stopwords=self.default_lang.split("-")[0])
        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        results, scores = self.retriever.retrieve(query_tokens, corpus=self.corpus, k=k)
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


class BM25QACorpusSolver(BM25CorpusSolver):
    def __init__(self, config=None):
        self.answers = {}
        super().__init__(config)

    def load_corpus(self, corpus: Dict):
        self.answers = corpus
        super().load_corpus(list(self.answers.keys()))

    def retrieve_from_corpus(self, query, k=1) -> str:
        for q in super().retrieve_from_corpus(query, k):
            LOG.debug(f"closest question in corpus: {q}")
            yield self.answers[q]

    def get_spoken_answer(self, query: str, context: Optional[dict] = None) -> str:
        if self.corpus is None:
            return None
        # Query the corpus
        answers = list(self.retrieve_from_corpus(query, k=self.config.get("n_answer", 1)))
        if answers:
            return ". ".join(answers)


class SquadQASolver(BM25QACorpusSolver):
    def __init__(self, config=None):
        super().__init__(config)
        self.load_squad_corpus()

    def load_squad_corpus(self):
        corpus = {}
        data = requests.get("https://github.com/chrischute/squad/raw/master/data/train-v2.0.json").json()
        for s in data["data"]:
            for p in s["paragraphs"]:
                for qa in p["qas"]:
                    if "question" in qa and qa["answers"]:
                        corpus[qa["question"]] = qa["answers"][0]["text"]
        self.load_corpus(corpus)
        LOG.info(f"Loaded and indexed {len(corpus)} question-answer pairs from SQuAD dataset")


class FreebaseQASolver(BM25QACorpusSolver):
    def __init__(self, config=None):
        super().__init__(config)
        self._load_freebase_dataset()

    def _load_freebase_dataset(self):
        # Convert FreebaseQA data into the required format
        corpus = {}
        data = requests.get("https://github.com/kelvin-jiang/FreebaseQA/raw/master/FreebaseQA-train.json").json()
        for qa in data["Questions"]:
            q = qa["ProcessedQuestion"]
            a = qa["Parses"][0]["Answers"][0]["AnswersName"][0]
            corpus[q] = a
        self.load_corpus(corpus)


if __name__ == "__main__":
    LOG.set_level("DEBUG")
    # Create your corpus here
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ]

    s = BM25CorpusSolver({})
    s.load_corpus(corpus)

    query = "does the fish purr like a cat?"
    print(s.spoken_answer(query))

    # 2024-07-19 20:03:29.979 - OVOS - ovos_plugin_manager.utils.config:get_plugin_config:40 - DEBUG - Loaded configuration: {'module': 'ovos-translate-plugin-server', 'lang': 'en-us'}
    # 2024-07-19 20:03:30.024 - OVOS - __main__:load_corpus:28 - DEBUG - indexed 4 documents
    # 2024-07-19 20:03:30.025 - OVOS - __main__:retrieve_from_corpus:37 - DEBUG - Rank 1 (score: 1.0584375858306885): a cat is a feline and likes to purr
    # 2024-07-19 20:03:30.025 - OVOS - __main__:retrieve_from_corpus:37 - DEBUG - Rank 2 (score: 0.481589138507843): a fish is a creature that lives in water and swims
    # a cat is a feline and likes to purr. a fish is a creature that lives in water and swims

    # hotpotqa dataset
    # data = requests.get("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json").json()
    # data = requests.get("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json").json()
    # for qa in data:
    #    corpus[qa["question"]] = qa["answer"]
    # len_hotpot = len(corpus) - len_squad - len_freebase
    # print(len_hotpot, "qa pairs imported from hotpotqa dataset")

    # s = BM25QACorpusSolver({})
    # s.load_corpus(corpus)

    s = FreebaseQASolver()
    query = "What is the capital of France"
    print("Query:", query)
    print("Answer:", s.spoken_answer(query))
    # 2024-07-19 22:31:09.468 - OVOS - __main__:load_corpus:60 - DEBUG - indexed 20357 documents
    # Query: What is the capital of France
    # 2024-07-19 22:31:09.468 - OVOS - __main__:retrieve_from_corpus:69 - DEBUG - Rank 1 (score: 5.996074199676514): what is the capital of france
    # 2024-07-19 22:31:09.469 - OVOS - __main__:retrieve_from_corpus:93 - DEBUG - closest question in corpus: what is the capital of france
    # Answer: paris

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
