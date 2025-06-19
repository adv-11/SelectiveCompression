"""
OrchestratorAgent

Listens for user messages (ignores agent‐tagged events)

Ingests into memory, retrieves context from all tiers

Builds a unified system prompt with Hot/Warm/Cold sections

Calls GPT4O-mini and emits the assistant’s reply
"""


import os
import logging
from langgraph import Graph, Agent, Message
import openai

from memory.memory_tier_manager import MemoryTierManager

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─── OpenAI Setup ──────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

# ─── OrchestratorAgent ─────────────────────────────────────────────────────────
class OrchestratorAgent(Agent):
    """
    Handles full message lifecycle:
     1) Ingest → memory
     2) Retrieve → multi-tier context
     3) Call LLM → gpt4o-mini
     4) Emit assistant reply
    """

    def __init__(self, name: str = "orchestrator_agent"):
        super().__init__(name=name)
        db_path = os.getenv("MEMORY_DB_PATH", "memory.db")
        self.memory = MemoryTierManager(db_path=db_path)
        logger.info(f"[{self.name}] Initialized with DB at {db_path}")

    @Agent.on_event("message")
    def handle_message(self, msg: Message):
        # Only handle direct user messages (no agent-generated ones)
        if msg.metadata.get("agent"):
            return

        user_text = msg.content.strip()
        logger.info(f"[{self.name}] Received user message: {user_text!r}")

        # ── 1) Ingest ─────────────────────────────────────────────────────────
        try:
            tier = self.memory.add(user_text)
            logger.info(f"[{self.name}] Stored in tier: {tier}")
        except Exception as e:
            logger.error(f"[{self.name}] Memory add error: {e}")

        # ── 2) Retrieve ────────────────────────────────────────────────────────
        hot_ctx  = self.memory.get_hot(top_k=3)
        warm_ctx = [f"- {item['summary']}" for item in self.memory.get_warm(user_text, top_k=2)]
        cold_items = self.memory.get_cold(user_text, top_k=2)
        cold_ctx = [f"- {item['content'][:200]}… (score {item['score']:.2f})"
                    for item in cold_items]

        # ── 3) Compose Prompt ──────────────────────────────────────────────────
        system_prompt = (
            "You are a helpful assistant.  \n"
            "Use the following context from memory tiers:\n\n"
            "Hot Memory (raw recent messages):\n" +
            "\n".join(f"- {text}" for text in hot_ctx) + "\n\n"
            "Warm Memory (summaries):\n" +
            "\n".join(warm_ctx) + "\n\n"
            "Cold Memory (archived context snippets):\n" +
            "\n".join(cold_ctx) + "\n\n"
            "Then answer the user’s new message."
        )

        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_text}
        ]

        # ── 4) LLM Call ────────────────────────────────────────────────────────
        try:
            resp = openai.ChatCompletion.create(
                model="gpt4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            assistant_text = resp.choices[0].message.content.strip()
            logger.info(f"[{self.name}] LLM responded.")
        except Exception as e:
            logger.error(f"[{self.name}] LLM error: {e}")
            assistant_text = "Sorry, I ran into an error while thinking."

        # ── 5) Emit Response ───────────────────────────────────────────────────
        self.emit(
            Message(
                content=assistant_text,
                metadata={"agent": self.name}
            )
        )


def main():
    graph = Graph()
    graph.add_agent(OrchestratorAgent())
    logger.info("Starting OrchestratorAgent event loop...")
    graph.run()


if __name__ == "__main__":
    main()
