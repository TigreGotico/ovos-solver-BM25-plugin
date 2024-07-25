from typing import List, Optional, Union, Tuple

from ovos_plugin_manager.templates.solvers import MultipleChoiceSolver, EvidenceSolver
from ovos_utils.log import LOG
from quebra_frases import sentence_tokenize

from ovos_bm25_solver.corpus import BM25CorpusSolver


class BM25MultipleChoiceSolver(MultipleChoiceSolver):
    """Select the best answer to a question from a list of options using BM25 ranking."""

    def rerank(self, query: str, options: List[str],
               lang: Optional[str] = None,
               return_index: bool = False) -> List[Tuple[float, Union[str, int]]]:
        """
        Rank the list of options based on their relevance to the query.

        Args:
            query (str): The query to rank options against.
            options (List[str]): A list of options to be ranked.
            lang (Optional[str]): Optional language code for translation.
            return_index (bool): Whether to return the index of the options instead of the option text.

        Returns:
            List[Tuple[float, Union[str, int]]]: A list of tuples containing the score and either the option text or its index.
        """
        bm25 = BM25CorpusSolver(internal_lang=self.default_lang,
                                translator=self.translator,
                                detector=self.detector,
                                enable_tx=self.enable_tx)
        bm25.load_corpus(options)
        ranked: List[Tuple[float, str]] = list(bm25.retrieve_from_corpus(query, lang=lang, k=len(options)))
        if return_index:
            ranked = [(r[0], options.index(r[1])) for r in ranked]
        return ranked


class BM25EvidenceSolverPlugin(EvidenceSolver):
    """Extract the best sentence from text that answers the question using BM25 algorithm."""

    def get_best_passage(self, evidence: str, question: str,
                         lang: Optional[str] = None) -> Optional[str]:
        """
        Extract the most relevant passage from the evidence that answers the question.

        Args:
            evidence (str): The text to search for the answer.
            question (str): The question to find an answer for.
            lang (Optional[str]): Optional language code for translation.

        Returns:
            Optional[str]: The best passage that answers the question, or None if no passage is found.
        """
        bm25 = BM25MultipleChoiceSolver(internal_lang=self.default_lang,
                                        translator=self.translator,
                                        detector=self.detector,
                                        enable_tx=self.enable_tx)
        sents = []
        for s in evidence.split("\n"):
            sents += sentence_tokenize(s)
        sents = [s.strip() for s in sents if s]
        return bm25.select_answer(question, sents, lang=lang)


if __name__ == "__main__":
    from ovos_bm25_solver.demo import BM25FreebaseQASolver, BM25SquadQASolver

    LOG.set_level("DEBUG")
    p = BM25MultipleChoiceSolver()
    a = p.rerank("what is the speed of light", [
        "very fast", "10m/s", "the speed of light is C"
    ], lang="en")
    print(a)
    # 2024-07-22 15:03:10.295 - OVOS - __main__:load_corpus:61 - DEBUG - indexed 3 documents
    # 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 1 (score: 0.7198746800422668): the speed of light is C
    # 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 2 (score: 0.0): 10m/s
    # 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 3 (score: 0.0): very fast
    # [(0.7198747, 'the speed of light is C'), (0.0, '10m/s'), (0.0, 'very fast')]

    a = p.select_answer("what is the speed of light", [
        "very fast", "10m/s", "the speed of light is C"
    ], lang="en")
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
    answer = solver.get_best_passage(evidence=text, question=query, lang="en")
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
    print(s.spoken_answer(query, lang="en"))

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

    s = BM25FreebaseQASolver()
    query = "What is the capital of France"
    print("Query:", query)
    print("Answer:", s.spoken_answer(query, lang="en"))
    # 2024-07-19 22:31:09.468 - OVOS - __main__:load_corpus:60 - DEBUG - indexed 20357 documents
    # Query: What is the capital of France
    # 2024-07-19 22:31:09.468 - OVOS - __main__:retrieve_from_corpus:69 - DEBUG - Rank 1 (score: 5.996074199676514): what is the capital of france
    # 2024-07-19 22:31:09.469 - OVOS - __main__:retrieve_from_corpus:93 - DEBUG - closest question in corpus: what is the capital of france
    # Answer: paris

    s = BM25SquadQASolver()
    query = "is there life on mars"
    print("Query:", query)
    print("Answer:", s.spoken_answer(query, lang="en"))
    # 2024-07-19 22:31:12.625 - OVOS - __main__:load_corpus:60 - DEBUG - indexed 86769 documents
    # 2024-07-19 22:31:12.625 - OVOS - __main__:load_squad_corpus:119 - INFO - Loaded and indexed 86769 question-answer pairs from SQuAD dataset
    # Query: is there life on mars
    # 2024-07-19 22:31:12.628 - OVOS - __main__:retrieve_from_corpus:69 - DEBUG - Rank 1 (score: 6.334013938903809): How is it postulated that Mars life might have evolved?
    # 2024-07-19 22:31:12.628 - OVOS - __main__:retrieve_from_corpus:93 - DEBUG - closest question in corpus: How is it postulated that Mars life might have evolved?
    # Answer: similar to Antarctic
