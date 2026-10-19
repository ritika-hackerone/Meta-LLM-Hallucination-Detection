"""
pipeline/rag_retriever.py
─────────────────────────
Retrieval-Augmented Generation (RAG) module.

Flow:
  1. Query → Wikipedia API → raw passages
  2. Encode passages with sentence-transformers
  3. Build FAISS index in memory
  4. Retrieve top-k passages most similar to the query
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import wikipediaapi
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger

import config


class RAGRetriever:
    """
    Retrieves factual evidence from Wikipedia using dense vector search.

    Attributes
    ----------
    model       : SentenceTransformer embedding model
    top_k       : number of passages to return
    wiki        : Wikipedia API client
    """

    def __init__(self, top_k: int = config.RAG_TOP_K):
        self.top_k   = top_k
        self.model   = SentenceTransformer(config.EMBEDDING_MODEL)
        self.wiki    = wikipediaapi.Wikipedia(
            user_agent="MetaLLMVerifier/1.0 (research project)",
            language="en",
        )
        logger.info(f"RAGRetriever ready | model={config.EMBEDDING_MODEL} | top_k={top_k}")

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Retrieve the most relevant passages for *query*.

        Returns
        -------
        passages : list of passage strings
        sources  : list of Wikipedia page titles
        """
        # 1. Fetch candidate pages
        raw_passages, sources = self._fetch_wikipedia(query)

        if not raw_passages:
            logger.warning(f"No Wikipedia passages found for: '{query}'")
            return [], []

        # 2. Embed passages + query
        passage_embeddings = self._encode(raw_passages)
        query_embedding    = self._encode([query])

        # 3. FAISS similarity search
        top_passages, top_sources = self._faiss_search(
            query_embedding, passage_embeddings, raw_passages, sources
        )

        logger.debug(f"Retrieved {len(top_passages)} passages for: '{query[:60]}...'")
        return top_passages, top_sources

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _fetch_wikipedia(self, query: str) -> Tuple[List[str], List[str]]:
        """Search Wikipedia and split pages into sentences."""
        passages: List[str] = []
        sources:  List[str] = []

        # Try the direct title first, then a keyword search
        search_terms = self._extract_keywords(query)

        for term in search_terms[:3]:          # max 3 page lookups
            page = self.wiki.page(term)
            if not page.exists():
                continue

            # Split into clean sentences
            text = page.summary + " " + page.text
            sents = self._split_sentences(text)

            for s in sents:
                if len(s.split()) >= 8:        # skip very short fragments
                    passages.append(s)
                    sources.append(page.title)

        return passages, sources

    def _encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype="float32")

    def _faiss_search(
        self,
        query_emb: np.ndarray,
        passage_embs: np.ndarray,
        passages: List[str],
        sources: List[str],
    ) -> Tuple[List[str], List[str]]:
        dim   = passage_embs.shape[1]
        index = faiss.IndexFlatIP(dim)         # inner-product on normalized = cosine
        index.add(passage_embs)

        k          = min(self.top_k, len(passages))
        _, indices = index.search(query_emb, k)

        top_passages = [passages[i] for i in indices[0]]
        top_sources  = [sources[i]  for i in indices[0]]
        return top_passages, top_sources

    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """Heuristically extract 1-3 Wikipedia search terms from the query."""
        # Remove question words and punctuation
        clean = re.sub(r"\b(what|who|where|when|why|how|is|are|was|were|does|do|did|can|the|a|an)\b",
                       "", query, flags=re.I)
        clean = re.sub(r"[?!.,;:]", "", clean).strip()

        # First term = original cleaned query (best Wikipedia title match)
        # Second = first 3 significant words
        words = [w for w in clean.split() if len(w) > 3]
        terms = [clean, " ".join(words[:3]), " ".join(words[:2])]
        return [t.strip() for t in terms if t.strip()]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        # Simple regex sentence splitter (avoids heavy nltk dependency)
        text   = re.sub(r"\s+", " ", text)
        sents  = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sents if s.strip()]









#2.
# """
# pipeline/rag_retriever.py
# ─────────────────────────
# Retrieval-Augmented Generation (RAG) module.

# Flow:
#   1. Query → Wikipedia API → raw passages
#   2. Encode passages with sentence-transformers
#   3. Build FAISS index in memory
#   4. Retrieve top-k passages most similar to the query
# """

# from __future__ import annotations

# import re
# from typing import List, Tuple

# import numpy as np
# import wikipediaapi
# import faiss
# from sentence_transformers import SentenceTransformer
# from loguru import logger

# import config


# class RAGRetriever:
#     """
#     Retrieves factual evidence from Wikipedia using dense vector search.

#     Attributes
#     ----------
#     model       : SentenceTransformer embedding model
#     top_k       : number of passages to return
#     wiki        : Wikipedia API client
#     """

#     def __init__(self, top_k: int = config.RAG_TOP_K):
#         self.top_k   = top_k
#         self.model   = SentenceTransformer(config.EMBEDDING_MODEL)
#         self.wiki    = wikipediaapi.Wikipedia(
#             user_agent="MetaLLMVerifier/1.0 (research project)",
#             language="en",
#         )
#         logger.info(f"RAGRetriever ready | model={config.EMBEDDING_MODEL} | top_k={top_k}")

#     # ──────────────────────────────────────────────────────────
#     # Public API
#     # ──────────────────────────────────────────────────────────

#     def retrieve(self, query: str) -> Tuple[List[str], List[str]]:
#         """
#         Retrieve the most relevant passages for *query*.

#         Returns
#         -------
#         passages : list of passage strings
#         sources  : list of Wikipedia page titles
#         """
#         # 1. Fetch candidate pages
#         raw_passages, sources = self._fetch_wikipedia(query)

#         if not raw_passages:
#             logger.warning(f"No Wikipedia passages found for: '{query}'")
#             return [], []

#         # 2. Embed passages + query
#         passage_embeddings = self._encode(raw_passages)
#         query_embedding    = self._encode([query])

#         # 3. FAISS similarity search
#         top_passages, top_sources = self._faiss_search(
#             query_embedding, passage_embeddings, raw_passages, sources
#         )

#         logger.debug(f"Retrieved {len(top_passages)} passages for: '{query[:60]}...'")
#         return top_passages, top_sources

#     # ──────────────────────────────────────────────────────────
#     # Private helpers
#     # ──────────────────────────────────────────────────────────

#     def _fetch_wikipedia(self, query: str) -> Tuple[List[str], List[str]]:
#         """Search Wikipedia and split pages into sentences."""
#         passages: List[str] = []
#         sources:  List[str] = []

#         # Try the direct title first, then a keyword search
#         search_terms = self._extract_keywords(query)

#         for term in search_terms[:3]:          # max 3 page lookups
#             page = self.wiki.page(term)
#             if not page.exists():
#                 continue

#             # Split into clean sentences
#             text = page.summary + " " + page.text
#             sents = self._split_sentences(text)

#             for s in sents:
#                 if len(s.split()) >= 8:        # skip very short fragments
#                     passages.append(s)
#                     sources.append(page.title)

#         return passages, sources

#     def _encode(self, texts: List[str]) -> np.ndarray:
#         embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
#         return np.array(embeddings, dtype="float32")

#     def _faiss_search(
#         self,
#         query_emb: np.ndarray,
#         passage_embs: np.ndarray,
#         passages: List[str],
#         sources: List[str],
#     ) -> Tuple[List[str], List[str]]:
#         dim   = passage_embs.shape[1]
#         index = faiss.IndexFlatIP(dim)         # inner-product on normalized = cosine
#         index.add(passage_embs)

#         k          = min(self.top_k, len(passages))
#         _, indices = index.search(query_emb, k)

#         top_passages = [passages[i] for i in indices[0]]
#         top_sources  = [sources[i]  for i in indices[0]]
#         return top_passages, top_sources

#     @staticmethod
#     def _extract_keywords(query: str) -> List[str]:
#         """Heuristically extract 1-3 Wikipedia search terms from the query."""
#         # Remove question words and punctuation
#         clean = re.sub(r"\b(what|who|where|when|why|how|is|are|was|were|does|do|did|can|the|a|an)\b",
#                        "", query, flags=re.I)
#         clean = re.sub(r"[?!.,;:]", "", clean).strip()

#         # First term = original cleaned query (best Wikipedia title match)
#         # Second = first 3 significant words
#         words = [w for w in clean.split() if len(w) > 3]
#         terms = [clean, " ".join(words[:3]), " ".join(words[:2])]
#         return [t.strip() for t in terms if t.strip()]

#     @staticmethod
#     def _split_sentences(text: str) -> List[str]:
#         # Simple regex sentence splitter (avoids heavy nltk dependency)
#         text   = re.sub(r"\s+", " ", text)
#         sents  = re.split(r"(?<=[.!?])\s+", text)
#         return [s.strip() for s in sents if s.strip()]






        


# """
# pipeline/rag_retriever.py
# ─────────────────────────
# Retrieval-Augmented Generation (RAG) module.

# Flow:
#   1. Query → Wikipedia API → raw passages
#   2. Encode passages with sentence-transformers
#   3. Build FAISS index in memory
#   4. Retrieve top-k passages most similar to the query
# """

# from __future__ import annotations

# import re
# from typing import List, Tuple

# import numpy as np
# import wikipediaapi
# import faiss
# from sentence_transformers import SentenceTransformer
# from loguru import logger

# import config


# class RAGRetriever:
#     """
#     Retrieves factual evidence from Wikipedia using dense vector search.

#     Attributes
#     ----------
#     model       : SentenceTransformer embedding model
#     top_k       : number of passages to return
#     wiki        : Wikipedia API client
#     """

#     def __init__(self, top_k: int = config.RAG_TOP_K):
#         self.top_k   = top_k
#         self.model   = SentenceTransformer(config.EMBEDDING_MODEL)
#         self.wiki    = wikipediaapi.Wikipedia(
#             user_agent="MetaLLMVerifier/1.0 (research project)",
#             language="en",
#         )
#         logger.info(f"RAGRetriever ready | model={config.EMBEDDING_MODEL} | top_k={top_k}")

#     # ──────────────────────────────────────────────────────────
#     # Public API
#     # ──────────────────────────────────────────────────────────

#     def retrieve(self, query: str) -> Tuple[List[str], List[str]]:
#         """
#         Retrieve the most relevant passages for *query*.

#         Returns
#         -------
#         passages : list of passage strings
#         sources  : list of Wikipedia page titles
#         """
#         # 1. Fetch candidate pages
#         raw_passages, sources = self._fetch_wikipedia(query)

#         if not raw_passages:
#             logger.warning(f"No Wikipedia passages found for: '{query}'")
#             return [], []

#         # 2. Embed passages + query
#         passage_embeddings = self._encode(raw_passages)
#         query_embedding    = self._encode([query])

#         # 3. FAISS similarity search
#         top_passages, top_sources = self._faiss_search(
#             query_embedding, passage_embeddings, raw_passages, sources
#         )

#         logger.debug(f"Retrieved {len(top_passages)} passages for: '{query[:60]}...'")
#         return top_passages, top_sources

#     # ──────────────────────────────────────────────────────────
#     # Private helpers
#     # ──────────────────────────────────────────────────────────

#     def _fetch_wikipedia(self, query: str) -> Tuple[List[str], List[str]]:
#         """Search Wikipedia and split pages into sentences."""
#         passages: List[str] = []
#         sources:  List[str] = []

#         # Try the direct title first, then a keyword search
#         search_terms = self._extract_keywords(query)

#         for term in search_terms[:3]:          # max 3 page lookups
#             page = self.wiki.page(term)
#             if not page.exists():
#                 continue

#             # Split into clean sentences
#             text = page.summary + " " + page.text
#             sents = self._split_sentences(text)

#             for s in sents:
#                 if len(s.split()) >= 8:        # skip very short fragments
#                     passages.append(s)
#                     sources.append(page.title)

#         return passages, sources

#     def _encode(self, texts: List[str]) -> np.ndarray:
#         embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
#         return np.array(embeddings, dtype="float32")

#     def _faiss_search(
#         self,
#         query_emb: np.ndarray,
#         passage_embs: np.ndarray,
#         passages: List[str],
#         sources: List[str],
#     ) -> Tuple[List[str], List[str]]:
#         dim   = passage_embs.shape[1]
#         index = faiss.IndexFlatIP(dim)         # inner-product on normalized = cosine
#         index.add(passage_embs)

#         k          = min(self.top_k, len(passages))
#         _, indices = index.search(query_emb, k)

#         top_passages = [passages[i] for i in indices[0]]
#         top_sources  = [sources[i]  for i in indices[0]]
#         return top_passages, top_sources

#     @staticmethod
#     def _extract_keywords(query: str) -> List[str]:
#         """Heuristically extract 1-3 Wikipedia search terms from the query."""
#         # Remove question words and punctuation
#         clean = re.sub(r"\b(what|who|where|when|why|how|is|are|was|were|does|do|did|can|the|a|an)\b",
#                        "", query, flags=re.I)
#         clean = re.sub(r"[?!.,;:]", "", clean).strip()

#         # First term = original cleaned query (best Wikipedia title match)
#         # Second = first 3 significant words
#         words = [w for w in clean.split() if len(w) > 3]
#         terms = [clean, " ".join(words[:3]), " ".join(words[:2])]
#         return [t.strip() for t in terms if t.strip()]

#     @staticmethod
#     def _split_sentences(text: str) -> List[str]:
#         # Simple regex sentence splitter (avoids heavy nltk dependency)
#         text   = re.sub(r"\s+", " ", text)
#         sents  = re.split(r"(?<=[.!?])\s+", text)
#         return [s.strip() for s in sents if s.strip()]
