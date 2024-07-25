from typing import List, Tuple, Union

from json_database import JsonStorageXDG
from ovos_plugin_manager.templates.embeddings import EmbeddingsDB
from ovos_plugin_manager.templates.embeddings import TextEmbeddingsStore
from ovos_utils.log import LOG
from ovos_utils.parse import MatchStrategy, fuzzy_match, match_all

from ovos_bm25_solver import BM25CorpusSolver


class JsonEmbeddingsDB(EmbeddingsDB):
    def __init__(self, path: str):
        super().__init__()
        self.corpus = JsonStorageXDG(name=path, subfolder="json_fake_embeddings")
        LOG.debug(f"JsonEmbeddingsDB index path: {self.corpus.path}")

    @property
    def documents(self):
        return list(self.corpus.keys())

    def add_embeddings(self, key: str, embedding: str) -> None:
        self.corpus[key] = embedding
        self.corpus.store()

    def delete_embedding(self, key: str) -> None:
        if key in self.corpus:
            self.corpus.pop(key)

    def get_embedding(self, key: str) -> str:
        return self.corpus.get(key)

    def query(self, embedding: str, top_k: int = 5) -> List[Tuple[str, float]]:
        return match_all(embedding, self.documents,
                         strategy=MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY)[:top_k]


class BM25TextEmbeddingsStore(TextEmbeddingsStore):

    def __init__(self, db: Union[EmbeddingsDB, str]):
        if isinstance(db, str):
            db = JsonEmbeddingsDB(path=db)
        if not isinstance(db, JsonEmbeddingsDB):
            raise ValueError("'db' should be a JsonFakeEmbeddingsDB")
        super().__init__(db)
        self.bm25 = BM25CorpusSolver()
        self.bm25.load_corpus(self.db.documents)

    def get_text_embeddings(self, text: str) -> str:
        return text

    def query(self, document: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Query the database for the top_k closest embeddings to the document.

        Args:
            document (str): The document to query.
            top_k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            List[Tuple[str, float]]: List of tuples containing the document and distance.
        """
        return [(txt, conf) for conf, txt in
                self.bm25.retrieve_from_corpus(document, k=top_k)]

    def distance(self, text_a: str, text_b: str,
                 metric: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY) -> float:
        """Calculate the distance between embeddings of two texts.

        Args:
            text_a (str): The first text.
            text_b (str): The second text.
            metric (MatchStrategy): DAMERAU_LEVENSHTEIN_SIMILARITY
        Returns:
            float: The calculated distance.
        """
        if not isinstance(metric, MatchStrategy):
            raise ValueError("'metric' must be a MatchStrategy for BM25 index")
        return fuzzy_match(text_a, text_b, strategy=metric)


if __name__ == "__main__":
    LOG.set_level("DEBUG")

    # Initialize the database
    db = JsonEmbeddingsDB("bm25_index")
    index = BM25TextEmbeddingsStore(db=db)

    # Add documents
    text = "hello world"
    text2 = "goodbye cruel world"
    index.add_document(text)
    index.add_document(text2)

    # query with fuzzy match
    results = db.query("the world", top_k=2)
    print(results)

    # query with bm25
    results = index.query("the world", top_k=2)
    print(results)

    # compare strings via fuzzy match - DAMERAU_LEVENSHTEIN_SIMILARITY
    print(index.distance(text, text2))
