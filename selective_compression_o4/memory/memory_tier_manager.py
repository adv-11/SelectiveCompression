"""
MemoryTierManager (memory_tier_manager.py)

On .add(text), routes via the AdaptiveControlSystem and:

Hot ➔ inserts raw text into hot_memory table.

Warm ➔ calls the LLM to summarize, then stores (summary, original) in warm_memory.

Cold ➔ calls the LLM embeddings endpoint, JSON‐serializes the vector, then stores (embedding_json, original) in cold_memory.

Spins up a single SQLite database (memory.db) with all three tables.

"""

import os
import sqlite3
import json
import logging
from datetime import datetime

import openai
import faiss
import numpy as np

from .adaptive_control_system import AdaptiveControlSystem

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─── OpenAI Setup ─────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")


class MemoryTierManager:
    """
    Manages hot, warm, and cold memory tiers in SQLite + FAISS:
      - Hot: raw text, immediate access
      - Warm: LLM‐summarized text + original
      - Cold: semantic embeddings + original; FAISS index for search
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        adaptive_control: AdaptiveControlSystem | None = None,
        embed_dim: int = 1536
    ):
        self.db_path = db_path
        self.adaptive_control = adaptive_control or AdaptiveControlSystem()
        self.embed_dim = embed_dim

        # SQLite connection (thread-safe)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

        # FAISS index for cold-tier
        self._init_faiss()

    def _create_tables(self):
        c = self.conn.cursor()
        # Hot memory
        c.execute("""
            CREATE TABLE IF NOT EXISTS hot_memory (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                content   TEXT    NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Warm memory (now also storing embedding_json for summaries)
        c.execute("""
            CREATE TABLE IF NOT EXISTS warm_memory (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                summary          TEXT    NOT NULL,
                original_content TEXT    NOT NULL,
                embedding_json   TEXT    NOT NULL,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Cold memory
        c.execute("""
            CREATE TABLE IF NOT EXISTS cold_memory (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_json   TEXT    NOT NULL,
                original_content TEXT    NOT NULL,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    # ─── FAISS SETUP ────────────────────────────────────────────────────────────
    def _init_faiss(self):
        # Create an IP index wrapped with ID map
        quant = faiss.IndexFlatIP(self.embed_dim)
        self.index = faiss.IndexIDMap(quant)

        # Load existing cold embeddings
        c = self.conn.cursor()
        c.execute("SELECT id, embedding_json FROM cold_memory")
        for row_id, embedding_json in c.fetchall():
            vec = np.array(json.loads(embedding_json), dtype="float32")
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self.index.add_with_ids(vec.reshape(1, -1), np.array([row_id], dtype="int64"))

        logger.info(f"FAISS index initialized with {self.index.ntotal} vectors")

    # ─── ADD METHODS ────────────────────────────────────────────────────────────
    def add(self, text: str) -> str:
        tier = self.adaptive_control.classify(text)
        logger.info(f"Storing text in tier '{tier}'")

        if tier == "hot":
            self._add_hot(text)
        elif tier == "warm":
            summary = self._summarize(text)
            emb_json = self._embed(summary)
            self._add_warm(summary, text, emb_json)
        else:  # cold
            emb_json = self._embed(text)
            self._add_cold(emb_json, text)

        return tier

    def _add_hot(self, text: str):
        c = self.conn.cursor()
        c.execute("INSERT INTO hot_memory(content) VALUES(?)", (text,))
        self.conn.commit()

    def _summarize(self, text: str) -> str:
        prompt = (
            "Summarize the following text in one concise sentence:\n"
            f"{text}"
        )
        resp = openai.ChatCompletion.create(
            model="gpt4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful summarizer."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=60,
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()

    def _embed(self, text: str) -> str:
        resp = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = resp["data"][0]["embedding"]
        return json.dumps(embedding)

    def _add_warm(self, summary: str, original: str, emb_json: str):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO warm_memory(summary, original_content, embedding_json) VALUES(?,?,?)",
            (summary, original, emb_json)
        )
        self.conn.commit()

    def _add_cold(self, embedding_json: str, original: str):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO cold_memory(embedding_json, original_content) VALUES(?,?)",
            (embedding_json, original)
        )
        row_id = c.lastrowid
        self.conn.commit()

        # Also add to FAISS index
        vec = np.array(json.loads(embedding_json), dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        self.index.add_with_ids(vec.reshape(1, -1), np.array([row_id], dtype="int64"))

    # ─── RETRIEVAL METHODS ───────────────────────────────────────────────────────
    def get_hot(self, top_k: int = 5) -> list[str]:
        """Return the last `top_k` raw messages from hot memory."""
        c = self.conn.cursor()
        c.execute(
            "SELECT content FROM hot_memory ORDER BY timestamp DESC LIMIT ?",
            (top_k,)
        )
        return [row[0] for row in c.fetchall()]

    def get_warm(self, top_k: int = 5) -> list[dict]:
        """
        Return the most recent `top_k` warm-memory summaries + contexts.
        (For now, ordered by recency; future: semantic match on summary embeddings.)
        """
        c = self.conn.cursor()
        c.execute(
            "SELECT summary, original_content FROM warm_memory "
            "ORDER BY timestamp DESC LIMIT ?",
            (top_k,)
        )
        return [
            {"summary": summary, "content": content}
            for summary, content in c.fetchall()
        ]

    def get_cold(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search in cold tier via FAISS.
        Returns top_k items with their similarity scores.
        """
        # Embed & normalize the query
        resp = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query
        )
        q_vec = np.array(resp["data"][0]["embedding"], dtype="float32")
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        # Search FAISS
        D, I = self.index.search(q_vec.reshape(1, -1), top_k)
        ids = I[0]
        scores = D[0]

        # Fetch contents
        results = []
        c = self.conn.cursor()
        for row_id, score in zip(ids, scores):
            if row_id == -1:
                continue
            c.execute(
                "SELECT original_content, timestamp FROM cold_memory WHERE id = ?",
                (int(row_id),)
            )
            row = c.fetchone()
            if not row:
                continue
            results.append({
                "id": int(row_id),
                "content": row[0],
                "timestamp": row[1],
                "score": float(score)
            })
        return results