from typing import List, Optional, Dict, Tuple

import bm25s
import requests
from ovos_plugin_manager.templates.solvers import QuestionSolver, MultipleChoiceSolver, EvidenceSolver
from ovos_utils.log import LOG
from quebra_frases import sentence_tokenize


class BM25CorpusSolver(QuestionSolver):
    enable_tx = False
    priority = 60
    METHODS = ["robertson", "lucene", "bm25l", "bm25+", "atire", "rank-bm25", "bm25-pt"]
    IDF_METHODS = ['robertson', 'lucene', 'atire', 'bm25l', 'bm25+']

    def __init__(self, config=None):
        config = config or {"lang": "en-us",
                            "min_conf": None,
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
            if self.config.get("min_conf"):
                if score >= self.config["min_conf"]:
                    yield doc, score
            else:
                yield doc, score

    def get_spoken_answer(self, query: str, context: Optional[dict] = None) -> str:
        if self.corpus is None:
            return None
        # Query the corpus
        answers = [a[0] for a in self.retrieve_from_corpus(query, k=self.config.get("n_answer", 1))]
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
        for q, score in super().retrieve_from_corpus(query, k):
            LOG.debug(f"closest question in corpus: {q}")
            yield self.answers[q]

    def get_spoken_answer(self, query: str, context: Optional[dict] = None) -> str:
        if self.corpus is None:
            return None
        # Query the corpus
        answers = list(self.retrieve_from_corpus(query, k=self.config.get("n_answer", 1)))
        if answers:
            return ". ".join(answers)


class BM25MultipleChoiceSolver(MultipleChoiceSolver):
    """select best answer to a question from a list of options """

    # plugin methods to override
    def rerank(self, query: str, options: List[str],
               context: Optional[dict] = None) -> List[Tuple[float, str]]:
        """
        rank options list, returning a list of tuples (score, text)
        """
        bm25 = BM25CorpusSolver()
        bm25.load_corpus(options)
        return [
            (score, doc) for doc, score in bm25.retrieve_from_corpus(query, k=len(options))
        ]


class BM25EvidenceSolverPlugin(EvidenceSolver):
    """extract best sentence from text that answers the question, using BM25 algorithm"""

    def get_best_passage(self, evidence, question, context=None):
        """
        evidence and question assured to be in self.default_lang
         returns summary of provided document
        """
        bm25 = BM25MultipleChoiceSolver()
        sents = []
        for s in evidence.split("\n"):
            sents += sentence_tokenize(s)
        sents = [s.strip() for s in sents if s]
        return bm25.select_answer(question, sents, context)


## Demo subclasses
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
    p = BM25MultipleChoiceSolver()
    a = p.rerank("what is the speed of light", [
        "very fast", "10m/s", "the speed of light is C"
    ])
    print(a)
    # 2024-07-22 15:03:10.295 - OVOS - __main__:load_corpus:61 - DEBUG - indexed 3 documents
    # 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 1 (score: 0.7198746800422668): the speed of light is C
    # 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 2 (score: 0.0): 10m/s
    # 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 3 (score: 0.0): very fast
    # [(0.7198747, 'the speed of light is C'), (0.0, '10m/s'), (0.0, 'very fast')]

    a = p.select_answer("what is the speed of light", [
        "very fast", "10m/s", "the speed of light is C"
    ])
    print(a)  # the speed of light is C

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
