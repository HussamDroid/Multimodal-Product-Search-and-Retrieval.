"""A small BM25 baseline with explicit English and Arabic normalization."""

from collections import Counter
import math
import re
import unicodedata

import numpy as np


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    text = text.translate(str.maketrans("أإآى", "اااي"))
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = [Counter(tokenize(doc)) for doc in documents]
        self.lengths = np.array([sum(doc.values()) for doc in self.documents], dtype=float)
        self.average = float(self.lengths.mean()) if len(self.lengths) else 1.0
        self.k1, self.b = k1, b
        df = Counter(term for doc in self.documents for term in doc)
        n = len(documents)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
                    for term, freq in df.items()}

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(len(self.documents), dtype=np.float32)
        length_factor = self.k1 * (1 - self.b + self.b * self.lengths / max(self.average, 1))
        for term in set(tokenize(query)):
            tf = np.array([doc.get(term, 0) for doc in self.documents])
            result += (self.idf.get(term, 0) * tf * (self.k1 + 1)
                       / np.maximum(tf + length_factor, 1e-9)).astype(np.float32)
        return result
