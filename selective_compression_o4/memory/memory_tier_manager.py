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
from .adaptive_control_system import AdaptiveControlSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure your OpenAI key is set
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")


class MemoryTierManager:
    """
    Manages hot, warm, and cold memory tiers in SQLite.
      - Hot: raw text, immediate access
      - Warm: LLM‐summarized text
      - Cold: semantic embeddings (JSON) + original text
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        adaptive_control: AdaptiveControlSystem | None = None
    ):
        self.db_path = db_path
        self.adaptive_control = adaptive_control or AdaptiveControlSystem()
        # SQLite connection; allow multi‐thread if your agents run concurrently
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()
        # Hot memory: raw text
        c.execute("""
            CREATE TABLE IF NOT EXISTS hot_memory (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                content       TEXT    NOT NULL,
                timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Warm memory: LLM‐generated summary + original
        c.execute("""
            CREATE TABLE IF NOT EXISTS warm_memory (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                summary           TEXT    NOT NULL,
                original_content  TEXT    NOT NULL,
                timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Cold memory: embedding JSON + original
        c.execute("""
            CREATE TABLE IF NOT EXISTS cold_memory (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_json    TEXT    NOT NULL,
                original_content  TEXT    NOT NULL,
                timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add(self, text: str) -> str:
        """
        Classify 'text' via AdaptiveControlSystem and
        store it in the appropriate SQLite table.
        Returns the tier it was stored in.
        """
        tier = self.adaptive_control.classify(text)
        logger.info(f"Storing text in tier '{tier}'")

        if tier == "hot":
            self._add_hot(text)
        elif tier == "warm":
            summary = self._summarize(text)
            self._add_warm(summary, text)
        else:  # cold
            embedding = self._embed(text)
            self._add_cold(embedding, text)

        return tier

    def _add_hot(self, text: str):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO hot_memory(content) VALUES(?)",
            (text,)
        )
        self.conn.commit()
        logger.debug("Inserted into hot_memory")

    def _summarize(self, text: str) -> str:
        """
        Uses gpt4o-mini to produce a one‐sentence summary.
        """
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
        summary = resp.choices[0].message.content.strip()
        logger.debug(f"Generated summary: {summary!r}")
        return summary

    def _add_warm(self, summary: str, original: str):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO warm_memory(summary, original_content) VALUES(?,?)",
            (summary, original)
        )
        self.conn.commit()
        logger.debug("Inserted into warm_memory")

    def _embed(self, text: str) -> str:
        """
        Generates an embedding via OpenAI and returns it as JSON.
        """
        resp = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = resp["data"][0]["embedding"]
        embedding_json = json.dumps(embedding)
        logger.debug("Generated embedding vector")
        return embedding_json

    def _add_cold(self, embedding_json: str, original: str):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO cold_memory(embedding_json, original_content) VALUES(?,?)",
            (embedding_json, original)
        )
        self.conn.commit()
        logger.debug("Inserted into cold_memory")
