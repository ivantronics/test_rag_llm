from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import math
import re

# Простой поисковик на основе TF-IDF с символными н-граммами по документу
# Если по какой-то причине sklearn не доступен, то используем фоллбек на python
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _HAVE_SKLEARN = True

except Exception:
    _HAVE_SKLEARN = False


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zа-я0-9ё\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class Document:
    doc_id: str
    content: str


class Retriever:
    def __init__(self, kb_dir: str):
        self.kb_dir = Path(kb_dir)
        self.docs: List[Document] = []
        self._fit()

    def _fit(self):
        for p in sorted(self.kb_dir.glob("*.md")):
            self.docs.append(
                Document(doc_id=p.name, content=p.read_text(encoding="utf-8")))
        if _HAVE_SKLEARN:
            self.vectorizer = TfidfVectorizer(preprocessor=_normalize)
            self.matrix = self.vectorizer.fit_transform(
                [d.content for d in self.docs])
        else:
            self.vocab = {}
            self.doc_tfs = []
            df = {}
            for d in self.docs:
                tokens = _normalize(d.content).split()
                tf = {}
                for t in tokens:
                    tf[t] = tf.get(t, 0) + 1
                self.doc_tfs.append(tf)
                for t in tf:
                    df[t] = df.get(t, 0) + 1
            self.idf = {}
            N = len(self.docs)
            for t, c in df.items():
                self.idf[t] = math.log((N + 1) / (c + 1)) + 1.0

    def _embed_query(self, q: str):
        if _HAVE_SKLEARN:
            return self.vectorizer.transform([q])
        else:
            tokens = _normalize(q).split()
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            return tf

    def _cosine(self, qv, doc_tf):
        q_weights = {}
        for t, f in qv.items():
            q_weights[t] = (f) * self.idf.get(t, 1.0)
        d_weights = {}
        for t, f in doc_tf.items():
            d_weights[t] = (f) * self.idf.get(t, 1.0)
        dot = 0.0
        for t, qw in q_weights.items():
            dot += qw * d_weights.get(t, 0.0)
        qnorm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
        dnorm = math.sqrt(sum(w * w for w in d_weights.values())) or 1.0
        return dot / (qnorm * dnorm)

    def retrieve(self, query: str, top_k: int = 3) -> List[
        Tuple[str, float, str]]:
        if not self.docs:
            return []
        if _HAVE_SKLEARN:
            qv = self._embed_query(query)
            sims = cosine_similarity(qv, self.matrix).ravel()
            pairs = list(zip([d.doc_id for d in self.docs], sims,
                             [d.content for d in self.docs]))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:top_k]
        else:
            qv = self._embed_query(query)
            scored = []
            for d, tf in zip(self.docs, self.doc_tfs):
                s = self._cosine(qv, tf)
                scored.append((d.doc_id, s, d.content))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
