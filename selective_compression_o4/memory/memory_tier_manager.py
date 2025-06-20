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
import time
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
    Manages hot, warm, and cold memory tiers:
      • Hot: raw text in SQLite
      • Warm: LLM summaries + summary embeddings in SQLite + FAISS
      • Cold: LLM embeddings in SQLite + FAISS
    Includes:
      – Retrieval APIs (get_hot, get_warm, get_cold)
      – Tier migration & pruning (garbage_collect)
      – Instrumentation → AdaptiveControlSystem
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        adaptive_control: AdaptiveControlSystem | None = None,
        embed_dim: int = 1536,
        stats_window: int = 20
    ):
        self.db_path = db_path
        self.adaptive_control = adaptive_control or AdaptiveControlSystem()
        self.embed_dim = embed_dim
        self.stats_window = stats_window

        # Usage stats for instrumentation
        self.usage_stats: dict[str, list[float]] = {
            "hot": [], "warm": [], "cold": []
        }

        # SQLite connection
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

        # FAISS indices
        self._init_cold_faiss()
        self._init_warm_faiss()

    # ─── TABLE CREATION ─────────────────────────────────────────────────────────
    def _create_tables(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS hot_memory (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                content   TEXT    NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS warm_memory (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                summary          TEXT    NOT NULL,
                original_content TEXT    NOT NULL,
                embedding_json   TEXT    NOT NULL,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS cold_memory (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_json   TEXT    NOT NULL,
                original_content TEXT    NOT NULL,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    # ─── FAISS INITIALIZATION ───────────────────────────────────────────────────
    def _init_cold_faiss(self):
        self.cold_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embed_dim))
        c = self.conn.cursor()
        c.execute("SELECT id, embedding_json FROM cold_memory")
        for row_id, emb_json in c.fetchall():
            vec = np.array(json.loads(emb_json), dtype="float32")
            norm = np.linalg.norm(vec)
            if norm > 0: vec /= norm
            self.cold_index.add_with_ids(vec.reshape(1, -1),
                                         np.array([row_id], dtype="int64"))
        logger.info(f"Cold FAISS index loaded ({self.cold_index.ntotal} vectors)")

    def _init_warm_faiss(self):
        self.warm_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embed_dim))
        c = self.conn.cursor()
        c.execute("SELECT id, embedding_json FROM warm_memory")
        for row_id, emb_json in c.fetchall():
            vec = np.array(json.loads(emb_json), dtype="float32")
            norm = np.linalg.norm(vec)
            if norm > 0: vec /= norm
            self.warm_index.add_with_ids(vec.reshape(1, -1),
                                         np.array([row_id], dtype="int64"))
        logger.info(f"Warm FAISS index loaded ({self.warm_index.ntotal} vectors)")

    # ─── ADD / INGEST ────────────────────────────────────────────────────────────
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
        prompt = f"Summarize the following text in one concise sentence:\n{text}"
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
        return json.dumps(resp["data"][0]["embedding"])

    def _add_warm(self, summary: str, original: str, emb_json: str):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO warm_memory(summary, original_content, embedding_json) VALUES(?,?,?)",
            (summary, original, emb_json)
        )
        row_id = c.lastrowid
        self.conn.commit()
        # Add to warm FAISS
        vec = np.array(json.loads(emb_json), dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0: vec /= norm
        self.warm_index.add_with_ids(vec.reshape(1, -1),
                                     np.array([row_id], dtype="int64"))

    def _add_cold(self, emb_json: str, original: str):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO cold_memory(embedding_json, original_content) VALUES(?,?)",
            (emb_json, original)
        )
        row_id = c.lastrowid
        self.conn.commit()
        vec = np.array(json.loads(emb_json), dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0: vec /= norm
        self.cold_index.add_with_ids(vec.reshape(1, -1),
                                     np.array([row_id], dtype="int64"))

    # ─── INSTRUMENTATION HELPER ────────────────────────────────────────────────
    def _record_latency(self, tier: str, ms: float):
        stats = self.usage_stats[tier]
        stats.append(ms)
        if len(stats) >= self.stats_window:
            # feed back all stats at once
            self.adaptive_control.adjust_thresholds(self.usage_stats)
            # reset
            for k in self.usage_stats: self.usage_stats[k].clear()

    # ─── RETRIEVAL ──────────────────────────────────────────────────────────────
    def get_hot(self, top_k: int = 5) -> list[str]:
        start = time.time()
        c = self.conn.cursor()
        c.execute(
            "SELECT content FROM hot_memory "
            "ORDER BY timestamp DESC LIMIT ?", (top_k,)
        )
        results = [r[0] for r in c.fetchall()]
        elapsed = (time.time() - start) * 1000
        self._record_latency("hot", elapsed)
        return results

    def get_warm(self, query: str, top_k: int = 5) -> list[dict]:
        # semantic search on summaries
        start = time.time()
        # embed & normalize
        resp = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query
        )
        qv = np.array(resp["data"][0]["embedding"], dtype="float32")
        norm = np.linalg.norm(qv)
        if norm > 0: qv /= norm

        D, I = self.warm_index.search(qv.reshape(1, -1), top_k)
        ids, scores = I[0], D[0]

        c = self.conn.cursor()
        out = []
        for rid, score in zip(ids, scores):
            if rid == -1: continue
            c.execute(
                "SELECT summary, original_content, timestamp "
                "FROM warm_memory WHERE id = ?", (int(rid),)
            )
            row = c.fetchone()
            if not row: continue
            out.append({
                "id":    int(rid),
                "summary": row[0],
                "content": row[1],
                "timestamp": row[2],
                "score": float(score)
            })
        elapsed = (time.time() - start) * 1000
        self._record_latency("warm", elapsed)
        return out

    def get_cold(self, query: str, top_k: int = 5) -> list[dict]:
        start = time.time()
        resp = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query
        )
        qv = np.array(resp["data"][0]["embedding"], dtype="float32")
        norm = np.linalg.norm(qv)
        if norm > 0: qv /= norm

        D, I = self.cold_index.search(qv.reshape(1, -1), top_k)
        ids, scores = I[0], D[0]

        c = self.conn.cursor()
        out = []
        for rid, score in zip(ids, scores):
            if rid == -1: continue
            c.execute(
                "SELECT original_content, timestamp FROM cold_memory WHERE id = ?",
                (int(rid),)
            )
            row = c.fetchone()
            if not row: continue
            out.append({
                "id":    int(rid),
                "content": row[0],
                "timestamp": row[1],
                "score": float(score)
            })
        elapsed = (time.time() - start) * 1000
        self._record_latency("cold", elapsed)
        return out

    # ─── TIER MIGRATION & GC ──────────────────────────────────────────────────
    def garbage_collect(
        self,
        hot_to_warm_age: float = 3600,        # in seconds
        warm_to_cold_age: float = 86400,
        cold_prune_age: float = 7 * 86400
    ):
        """
        - Migrate hot → warm if older than hot_to_warm_age
        - Migrate warm → cold if older than warm_to_cold_age
        - Prune cold entries older than cold_prune_age
        """
        now = datetime.utcnow()
        c = self.conn.cursor()

        # Hot → Warm
        days = hot_to_warm_age / 86400
        c.execute(
            "SELECT id, content FROM hot_memory "
            "WHERE (julianday('now') - julianday(timestamp)) > ?",
            (days,)
        )
        for hid, content in c.fetchall():
            summary = self._summarize(content)
            emb_json = self._embed(summary)
            self._add_warm(summary, content, emb_json)
            c.execute("DELETE FROM hot_memory WHERE id = ?", (hid,))

        # Warm → Cold
        days2 = warm_to_cold_age / 86400
        c.execute(
            "SELECT id, original_content FROM warm_memory "
            "WHERE (julianday('now') - julianday(timestamp)) > ?",
            (days2,)
        )
        for wid, original in c.fetchall():
            emb_json = self._embed(original)
            self._add_cold(emb_json, original)
            c.execute("DELETE FROM warm_memory WHERE id = ?", (wid,))

        # Prune Cold
        days3 = cold_prune_age / 86400
        c.execute(
            "SELECT id FROM cold_memory "
            "WHERE (julianday('now') - julianday(timestamp)) > ?",
            (days3,)
        )
        for (cid,) in c.fetchall():
            self.cold_index.remove_ids(np.array([cid], dtype="int64"))
            c.execute("DELETE FROM cold_memory WHERE id = ?", (cid,))

        self.conn.commit()
        logger.info("Completed garbage collection & tier migration")




# ─── Singleton Accessor ────────────────────────────────────────────────────────
_shared_manager: MemoryTierManager | None = None

def get_memory_manager() -> MemoryTierManager:
    """
    Returns a single shared MemoryTierManager across all agents.
    """
    global _shared_manager
    if _shared_manager is None:
        _shared_manager = MemoryTierManager()
    return _shared_manager