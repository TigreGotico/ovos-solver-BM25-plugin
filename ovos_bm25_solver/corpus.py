from typing import List, Optional, Tuple, Iterable

import bm25s
from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_plugin_manager.templates.solvers import QACorpusSolver, CorpusSolver
from ovos_utils.log import LOG


class BM25CorpusSolver(CorpusSolver):
    METHODS = ["robertson", "lucene", "bm25l", "bm25+", "atire", "rank-bm25", "bm25-pt"]
    IDF_METHODS = ['robertson', 'lucene', 'atire', 'bm25l', 'bm25+']

    def __init__(self, config=None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs):
        config = config or {"min_conf": 0.0,
                            "n_answer": 1,
                            "method": None,
                            "idf_method": None}
        super().__init__(config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)
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
        stopwords = self.default_lang.split("-")[0]
        if stopwords != "en":
            stopwords = []

        corpus_tokens = bm25s.tokenize(corpus, stopwords=stopwords)
        # index the corpus
        self.retriever.index(corpus_tokens)
        LOG.debug(f"indexed {len(corpus)} documents")

    def query(self, query: str, lang: Optional[str], k: int = 3) -> Iterable[Tuple[str, float]]:
        lang = lang or self.default_lang
        stopwords = lang.split("-")[0]
        if stopwords != "en":
            stopwords = []
        query_tokens = bm25s.tokenize(query, stopwords=stopwords)
        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        results, scores = self.retriever.retrieve(query_tokens, corpus=self.corpus, k=k)
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            yield doc, score


class BM25QACorpusSolver(QACorpusSolver, BM25CorpusSolver):
    """"""
