import os
import logging

from langgraph import Graph, Agent, Message
import openai

from memory.memory_tier_manager import MemoryTierManager

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

# ─── ChatAgent ────────────────────────────────────────────────────────────────
class ChatAgent(Agent):
    """
    A LangGraph agent that:
      1. Stores each incoming user turn in the memory tiers.
      2. (Later) could retrieve relevant context.
      3. Forwards the message + context to gpt4o-mini and emits the response.
    """

    def __init__(self, name: str = "chat_agent"):
        super().__init__(name=name)
        # Initialize the MemoryTierManager (uses memory.db by default)
        db_path = os.getenv("MEMORY_DB_PATH", "memory.db")
        self.memory = MemoryTierManager(db_path=db_path)
        logger.info(f"Initialized ChatAgent with memory DB at {db_path}")

    @Agent.on_event("message")
    def handle_message(self, msg: Message):
        user_text = msg.content.strip()
        logger.info(f"[{self.name}] Received user message: {user_text!r}")

        # ── 1. STORE the user turn in memory ──────────────────────────────
        try:
            tier = self.memory.add(user_text)
            logger.info(f"[{self.name}] Stored message in '{tier}' tier")
        except Exception as e:
            logger.error(f"[{self.name}] Memory storage error: {e}")

        # ── 2. (Future) RETRIEVE context from memory here ─────────────────
        # e.g. context = self.memory.get_relevant(user_text)

        # ── 3. CALL the LLM ───────────────────────────────────────────────
        try:
            response = openai.ChatCompletion.create(
                model="gpt4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": user_text}
                ],
                temperature=0.7,
                max_tokens=512
            )
            assistant_text = response.choices[0].message.content.strip()
            logger.info(f"[{self.name}] LLM responded: {assistant_text!r}")
        except Exception as e:
            logger.error(f"[{self.name}] LLM call failed: {e}")
            assistant_text = "Sorry, I encountered an error while thinking."

        # ── 4. EMIT the LLM’s reply back into the graph ────────────────────
        self.emit(
            Message(content=assistant_text, metadata={"agent": self.name})
        )


def main():
    """
    Entrypoint: build the graph, register the ChatAgent, and start listening.
    """
    graph = Graph()
    agent = ChatAgent()
    graph.add_agent(agent)
    logger.info("Starting LangGraph event loop...")
    graph.run()  # Blocks here


if __name__ == "__main__":
    main()
