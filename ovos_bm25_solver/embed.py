from typing import List, Tuple, Union, Optional, Dict

from json_database import JsonStorageXDG
from ovos_utils.log import LOG
from ovos_utils.parse import MatchStrategy, fuzzy_match, match_all

from ovos_bm25_solver import BM25CorpusSolver
from ovos_plugin_manager.templates.embeddings import EmbeddingsDB, TextEmbeddingsStore


class JsonEmbeddingsDB(EmbeddingsDB):
    """An implementation of EmbeddingsDB using JSON-based storage for embeddings."""

    def __init__(self, path: str):
        """Initialize the JsonEmbeddingsDB.

        Args:
            path (str): The path to the JSON storage.
        """
        super().__init__()
        self.corpus = JsonStorageXDG(name=path, subfolder="json_fake_embeddings")
        LOG.debug(f"JsonEmbeddingsDB index path: {self.corpus.path}")

    @property
    def documents(self) -> List[str]:
        """List of document keys in the database.

        Returns:
            List[str]: A list of document keys.
        """
        return list(self.corpus.keys())

    def add_embeddings(self, key: str, _, metadata: Optional[Dict[str, any]] = None) -> str:
        """Add or update an embedding in the database.

        Args:
            key (str): The unique key for the embedding.
            metadata (Optional[Dict[str, any]]): Optional metadata associated with the embedding.

        Returns:
            str: The key of the added or updated embedding.
        """
        self.corpus[key] = metadata or {}
        self.corpus.store()
        return key

    def delete_embedding(self, key: str) -> Optional[Dict[str, any]]:
        """Delete an embedding from the database.

        Args:
            key (str): The unique key for the embedding to delete.

        Returns:
            Optional[Dict[str, any]]: The metadata of the deleted embedding, or None if the key was not found.
        """
        if key in self.corpus:
            return self.corpus.pop(key)

    def get_embeddings(self, key: str) -> Optional[Dict[str, any]]:
        """Retrieve an embedding from the database.

        Args:
            key (str): The unique key for the embedding to retrieve.

        Returns:
            Optional[Dict[str, any]]: The metadata of the embedding, or None if the key was not found.
        """
        return self.corpus.get(key)

    def query(self, embedding: str, top_k: int = 5, return_metadata: bool = False) -> List[Tuple[str, float]]:
        """Query the database for the closest embeddings to the given query embedding.

        Args:
            embedding (str): The query embedding to match.
            top_k (int, optional): The number of top results to return. Defaults to 5.
            return_metadata (bool, optional): Whether to include metadata in the results. Defaults to False.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing the matched document and the similarity score.
        """
        matches: List[Tuple[str, float]] = match_all(embedding, self.documents,
                                                     strategy=MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY)[:top_k]
        if return_metadata:
            return [(k, v, self.get_embeddings(k)) for k, v in matches]
        return matches


class BM25TextEmbeddingsStore(TextEmbeddingsStore):
    """A text embeddings store using BM25 for document retrieval."""

    def __init__(self, db: Union[EmbeddingsDB, str]):
        """Initialize the BM25TextEmbeddingsStore.

        Args:
            db (Union[EmbeddingsDB, str]): An instance of EmbeddingsDB or a path to initialize JsonEmbeddingsDB.

        Raises:
            ValueError: If `db` is not an instance of JsonEmbeddingsDB.
        """
        if isinstance(db, str):
            db = JsonEmbeddingsDB(path=db)
        super().__init__(db)
        if not isinstance(self.db, JsonEmbeddingsDB):
            raise ValueError("'db' should be a JsonEmbeddingsDB instance")

    def get_text_embeddings(self, text: str) -> str:
        """Convert text to its corresponding embeddings.

        Args:
            text (str): The input text to be converted.

        Returns:
            str: The text itself (no transformation applied in this implementation).
        """
        return text

    def query(self, document: str, top_k: int = 5, return_metadata: bool = False) -> List[Tuple[str, float, Optional[Dict[str, any]]]]:
        """Query the database for the top_k closest documents to the given document.

        Args:
            document (str): The document to query.
            top_k (int, optional): The number of top results to return. Defaults to 5.
            return_metadata (bool, optional): Whether to include metadata in the results. Defaults to False.

        Returns:
            List[Tuple[str, float, Optional[Dict[str, any]]]]: A list of tuples containing the document, score, and optionally metadata.
        """

        bm25 = BM25CorpusSolver()
        bm25.load_corpus(self.db.documents)
        if return_metadata:
            return [(txt, conf, self.db.get_embeddings(txt)) for conf, txt in
                    bm25.retrieve_from_corpus(document, k=top_k)]
        return [(txt, conf) for conf, txt in
                bm25.retrieve_from_corpus(document, k=top_k)]

    def distance(self, text_a: str, text_b: str,
                 metric: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY) -> float:
        """Calculate the distance between two texts using a specified metric.

        Args:
            text_a (str): The first text.
            text_b (str): The second text.
            metric (MatchStrategy): The match strategy to use. Defaults to DAMERAU_LEVENSHTEIN_SIMILARITY.

        Returns:
            float: The calculated distance between the texts.

        Raises:
            ValueError: If `metric` is not an instance of MatchStrategy.
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

    # Query with fuzzy match
    results = db.query("the world", top_k=2)
    print(results)

    # Query with BM25
    results = index.query("the world", top_k=2)
    print(results)

    # Compare strings via fuzzy match - DAMERAU_LEVENSHTEIN_SIMILARITY
    print(index.distance(text, text2))
