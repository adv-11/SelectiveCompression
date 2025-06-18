Date: 18th June

### AdaptiveControlSystem

Classifies any text into hot/warm/cold using GPT4O-mini.

Exposes methods for classification and threshold adjustment.

#### MemoryTierManager

Storage:

Hot: raw text in SQLite table.

Warm: LLM-generated summary + its embedding + original text.

Cold: LLM-generated embedding + original text, with FAISS index for semantic lookup.

#### Retrieval:

get_hot(top_k): returns most recent hot entries.

get_warm(top_k): returns most recent warm summaries & contexts.

get_cold(query, top_k): semantic FAISS search over cold embeddings.

#### ChatAgent

On each incoming user message:

Calls MemoryTierManager.add() to store the turn.

(Future) will retrieve relevant context.

Forwards the user message to GPT4O-mini and emits the response.
