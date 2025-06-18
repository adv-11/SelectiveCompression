import os
import logging
from langgraph import Graph, Agent, Message  # assumes langgraph provides these primitives
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure your OpenAI key is set in the environment
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")


class ChatAgent(Agent):
    """
    A simple LangGraph agent that:
      - Listens for incoming user messages
      - Forwards each message to gpt4o-mini
      - Emits the LLM response back into the graph
    """

    def __init__(self, name: str = "chat_agent"):
        super().__init__(name=name)
        logger.info(f"Initializing agent {self.name} with model=gpt4o-mini")

    @Agent.on_event("message")  # decorate to handle any Message events
    def handle_message(self, msg: Message):
        user_text = msg.content
        logger.info(f"[{self.name}] Received message: {user_text!r}")

        # Call GPT4O Mini
        try:
            resp = openai.ChatCompletion.create(
                model="gpt4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": user_text}
                ],
                temperature=0.7,
                max_tokens=512
            )
            generated = resp.choices[0].message.content
            logger.info(f"[{self.name}] LLM responded: {generated!r}")
        except Exception as e:
            logger.error(f"[{self.name}] OpenAI request failed: {e}")
            generated = "Sorry, I ran into an error while thinking."

        # Emit the response back into the graph
        self.emit(Message(content=generated, metadata={"agent": self.name}))


def main():
    """
    Entrypoint: builds a Graph, attaches the ChatAgent, and starts the loop.
    """
    graph = Graph()

    # Instantiate and register our agent
    chat_agent = ChatAgent()
    graph.add_agent(chat_agent)

    logger.info("Starting LangGraph event loop...")
    graph.run()  # blocks, listening for incoming Message events


if __name__ == "__main__":
    main()
