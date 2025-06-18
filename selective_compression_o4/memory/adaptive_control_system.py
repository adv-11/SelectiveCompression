"""
AdaptiveControlSystem (adaptive_control_system.py)

Uses gpt4o-mini for live classification of any text into hot/warm/cold.

Exposes .classify(text), .assign_tier(item), and a feedback‐driven .adjust_thresholds() method.
"""

import os
import logging

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


class AdaptiveControlSystem:
    """
    Orchestrates memory-tier decisions:
      - classify(text) → 'hot' | 'warm' | 'cold'
      - assign_tier(item) → routes arbitrary dicts via classify()
      - adjust_thresholds(usage_stats) → tune ms thresholds per tier
    """

    def __init__(
        self,
        hot_threshold_ms: float = 100.0,
        warm_threshold_ms: float = 500.0,
        cold_threshold_ms: float = 2000.0,
    ):
        # retrieval SLA targets in milliseconds
        self.hot_threshold_ms = hot_threshold_ms
        self.warm_threshold_ms = warm_threshold_ms
        self.cold_threshold_ms = cold_threshold_ms
        logger.info(
            f"AdaptiveControlSystem initialized with thresholds: "
            f"hot={self.hot_threshold_ms}ms, "
            f"warm={self.warm_threshold_ms}ms, "
            f"cold={self.cold_threshold_ms}ms"
        )

    def classify(self, text: str) -> str:
        """
        Uses GPT4O-mini to classify a text snippet into one of:
        'hot', 'warm', or 'cold'.
        """
        system_prompt = (
            "You are a smart router that assigns pieces of conversation "
            "to one of three memory tiers based on importance and recency:\n"
            "- hot: needs immediate access (e.g. current turn)\n"
            "- warm: recent but secondary context\n"
            "- cold: archived, rarely used context\n"
            "Respond with exactly one word: hot, warm, or cold."
        )
        user_prompt = f'Classify the following text:\n"""{text}"""'

        logger.debug("Sending classification request to gpt4o-mini")
        resp = openai.ChatCompletion.create(
            model="gpt4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1,  # force single-word response
        )
        tier = resp.choices[0].message.content.strip().lower()
        if tier not in {"hot", "warm", "cold"}:
            logger.warning(f"Unexpected tier '{tier}', defaulting to 'warm'")
            return "warm"
        logger.info(f"Classified text as '{tier}'")
        return tier

    def assign_tier(self, item: dict) -> str:
        """
        Takes a dict with at least a 'text' field and returns
        the assigned memory tier.
        """
        text = item.get("text", "")
        return self.classify(text)

    def adjust_thresholds(self, usage_stats: dict[str, list[float]]) -> None:
        """
        Given observed retrieval latencies per tier in milliseconds:
          usage_stats = {
            "hot": [t1, t2, ...],
            "warm": [...],
            "cold": [...],
          }
        Recomputes thresholds as 1.2x the average observed time.
        """
        for tier in ("hot", "warm", "cold"):
            times = usage_stats.get(tier, [])
            if not times:
                logger.debug(f"No data for {tier}, keeping existing threshold")
                continue
            avg = sum(times) / len(times)
            new_threshold = avg * 1.2
            setattr(self, f"{tier}_threshold_ms", new_threshold)
            logger.info(
                f"Adjusted {tier} threshold to {new_threshold:.1f}ms "
                f"(avg observed {avg:.1f}ms)"
            )
