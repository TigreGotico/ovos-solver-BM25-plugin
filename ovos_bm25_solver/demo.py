from typing import Optional, Dict, Any

import requests
from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_utils.log import LOG

from ovos_bm25_solver.corpus import BM25QACorpusSolver


## Demo subclasses
class BM25SquadQASolver(BM25QACorpusSolver):
    """
    A QA solver that uses the BM25 algorithm and loads data from the SQuAD dataset.

    Attributes:
        internal_lang (str): Language code used internally.
    """

    def __init__(self, config: Optional[dict] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 *args, **kwargs):
        """
        Initialize the BM25SquadQASolver with optional configurations.

        Args:
            config (Optional[dict]): Configuration dictionary. Defaults to {"n_answer": 1}.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority level for the solver. Defaults to 50.
            enable_tx (bool): Whether to enable translation. Defaults to False.
            enable_cache (bool): Whether to enable caching. Defaults to False.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        internal_lang = "en-us"
        config = config or {"n_answer": 1}
        super().__init__(config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)

    def load_squad_corpus(self):
        """
        Load and index the SQuAD dataset.

        This method fetches the SQuAD dataset from the specified URL, processes it, and
        loads it into the BM25 corpus.
        """
        corpus: Dict[str, str] = {}
        response = requests.get("https://github.com/chrischute/squad/raw/master/data/train-v2.0.json")
        data = response.json()
        for s in data["data"]:
            for p in s["paragraphs"]:
                for qa in p["qas"]:
                    if "question" in qa and qa["answers"]:
                        corpus[qa["question"]] = qa["answers"][0]["text"]
        self.load_corpus(corpus)
        LOG.info(f"Loaded and indexed {len(corpus)} question-answer pairs from SQuAD dataset")


class BM25FreebaseQASolver(BM25QACorpusSolver):
    """
    A QA solver that uses the BM25 algorithm and loads data from the FreebaseQA dataset.

    Attributes:
        internal_lang (str): Language code used internally.
    """

    def __init__(self, config: Optional[dict] = None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 *args, **kwargs):
        """
        Initialize the BM25FreebaseQASolver with optional configurations.

        Args:
            config (Optional[dict]): Configuration dictionary. Defaults to {"n_answer": 1}.
            translator (Optional[LanguageTranslator]): Optional language translator.
            detector (Optional[LanguageDetector]): Optional language detector.
            priority (int): Priority level for the solver. Defaults to 50.
            enable_tx (bool): Whether to enable translation. Defaults to False.
            enable_cache (bool): Whether to enable caching. Defaults to False.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        internal_lang = "en-us"
        config = config or {"n_answer": 1}
        super().__init__(config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)
        self._load_freebase_dataset()

    def _load_freebase_dataset(self):
        """
        Load and index the FreebaseQA dataset.

        This method fetches the FreebaseQA dataset from the specified URL, processes it, and
        loads it into the BM25 corpus.
        """
        corpus: Dict[str, str] = {}
        response = requests.get("https://github.com/kelvin-jiang/FreebaseQA/raw/master/FreebaseQA-train.json")
        data = response.json()
        for qa in data["Questions"]:
            q = qa["ProcessedQuestion"]
            a = qa["Parses"][0]["Answers"][0]["AnswersName"][0]
            corpus[q] = a
        self.load_corpus(corpus)
