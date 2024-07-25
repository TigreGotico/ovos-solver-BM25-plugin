from typing import Optional

import requests
from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_utils.log import LOG

from ovos_bm25_solver.corpus import BM25QACorpusSolver


## Demo subclasses
class BM25SquadQASolver(BM25QACorpusSolver):
    def __init__(self, config=None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 *args, **kwargs):
        internal_lang = "en-us"
        config = config or {"n_answer": 1}
        super().__init__(config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)

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


class BM25FreebaseQASolver(BM25QACorpusSolver):
    def __init__(self, config=None,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None,
                 priority: int = 50,
                 enable_tx: bool = False,
                 enable_cache: bool = False,
                 *args, **kwargs):
        internal_lang = "en-us"
        config = config or {"n_answer": 1}
        super().__init__(config, translator, detector, priority,
                         enable_tx, enable_cache, internal_lang,
                         *args, **kwargs)
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
