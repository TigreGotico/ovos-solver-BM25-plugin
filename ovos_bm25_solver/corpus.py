from typing import List, Optional, Tuple, Iterable

import bm25s
from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_plugin_manager.templates.solvers import QACorpusSolver, CorpusSolver
from ovos_utils.log import LOG


class BM25CorpusSolver(CorpusSolver):
    """
    A corpus solver that uses the BM25 algorithm for information retrieval.

    Attributes:
        METHODS (List[str]): List of supported BM25 methods.
        IDF_METHODS (List[str]): List of supported IDF methods.
    """

    METHODS = ["robertson", "lucene", "bm25l", "bm25+", "atire", "rank-bm25", "bm25-pt"]
    IDF_METHODS = ['robertson', 'lucene', 'atire', 'bm25l', 'bm25+']

    def __init__(self, config: Optional[dict] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 internal_lang: Optional[str] = None,
                 *args, **kwargs):
        """
        Initialize the BM25CorpusSolver with optional configurations.

        Args:
            config (Optional[dict]): Configuration dictionary. Default is None.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority level for the solver. Default is 50.
            enable_tx (bool): Whether to enable translation. Default is False.
            enable_cache (bool): Whether to enable caching. Default is False.
            internal_lang (Optional[str]): Language code for internal processing. Default is None.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
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
    def method(self) -> Optional[str]:
        """
        Get the BM25 method from configuration.

        Returns:
            Optional[str]: The BM25 method or None if not configured.
        """
        m = self.config.get("method")
        if m is None:
            return None
        if m not in self.METHODS:
            LOG.warning(f"{m} is not a valid method, choose one of {self.METHODS}")
            m = None
        return m

    @property
    def idf_method(self) -> Optional[str]:
        """
        Get the IDF method from configuration.

        Returns:
            Optional[str]: The IDF method or None if not configured. Defaults to "lucene" if invalid.
        """
        m = self.config.get("idf_method")
        if m is None:
            return None
        if m not in self.IDF_METHODS:
            LOG.warning(f"{m} is not a valid method, choose one of {self.IDF_METHODS}")
            m = "lucene"
        return m

    def load_corpus(self, corpus: List[str]):
        """
        Load and index the given corpus using the BM25 algorithm.

        Args:
            corpus (List[str]): A list of documents to be indexed.
        """
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

    def query(self, query: str, lang: Optional[str] = None, k: int = 3) -> Iterable[Tuple[str, float]]:
        """
        Query the indexed corpus and yield the top-k results.

        Args:
            query (str): The query string to search for.
            lang (Optional[str]): Optional language code for tokenization.
            k (int): The number of top results to return. Defaults to 3.

        Yields:
            Tuple[str, float]: Tuples containing the document ID and its score.
        """
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
    """
    A QA corpus solver that combines BM25 retrieval with question answering capabilities.
    """
