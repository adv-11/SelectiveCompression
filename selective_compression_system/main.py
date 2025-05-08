from interface.llm_interface import LLMInterface
from modules.entity_extractor import EntityExtractor
from modules.compressor import CompressorModule
from modules.relevance import RelevanceEvaluator
from core.memory_manager import MemoryManager
from modules.retriever import RetrievalEngine
from interface.integration import IntegrationLayer
from core.memory import MemorySegment, MemoryTier
import time
from uuid import uuid4


# Initialize the required components
entity_extractor = EntityExtractor()
llm_interface = LLMInterface()  # Removed the model_name argument

# Pass the required arguments to CompressorModule
compressor = CompressorModule(llm_interface=llm_interface, entity_extractor=entity_extractor)

# Initialize the RelevanceEvaluator with the required arguments
relevance_evaluator = RelevanceEvaluator(
    embedding_model=None,  # Replace `None` with the actual embedding model if available
    entity_extractor=entity_extractor
)

# Initialize the MemoryManager with the corrected compressor
memory_manager = MemoryManager(
    hot_size=4000, 
    warm_size=16000, 
    cold_size=64000,
    compressor=compressor, 
    relevance_evaluator=relevance_evaluator
)


# Text segment to compress


text = """
During the meeting on March 15th, we discussed the Q1 results and projected targets for Q2.
The marketing team presented their campaign results, showing a 12% increase in engagement compared to last quarter.
John Smith suggested we should increase our social media budget by 15%, while Mary Johnson argued for expanding our email marketing efforts instead.
The development team is on track with the new product features, expecting to launch version 2.5 by the end of April.
Customer satisfaction scores improved to 8.7/10, up from 8.2/10 in the previous quarter.
"""


print (' Text segment to compress: ', text)
print (' ---------------------------------------------------------------------------------------------- ')


# Create a memory segment
segment = MemorySegment(
    content=text,
    id="segment-1",
    creation_time=time.time(),
    importance=0.8,
    compression_level=0
)

# Add to hot memory
memory_manager.hot_memory.add_segment(segment)

# Simulate memory pressure and force compression
compressed_segment = memory_manager.compressor.compress(segment, level=1)

print(f"Original size: {len(segment.content)}")
print(f"Compressed size: {len(compressed_segment.content)}")
print(f"Compression ratio: {len(segment.content) / len(compressed_segment.content):.2f}x")
print(f"\nCompressed content:\n{compressed_segment.content}")


print (' ---------------------------------------------------------------------------------------------- ')

# Example of retrieving and decompressing a segment
retrieval_engine = RetrievalEngine(memory_manager, relevance_evaluator, llm_interface)

# Current context and query
current_context = "We need to review our marketing budget allocation. What did we discuss about it previously?"

# Find relevant segments
relevant_segments = retrieval_engine.identify_relevant_segments(current_context)

# Decompress the most relevant segment
if relevant_segments:
    decompressed = retrieval_engine.decompress_segment(relevant_segments[0])
    print(f"Retrieved and decompressed:\n{decompressed.content}")
else:
    print("No relevant segments found")


# Key methods in the Memory Manager API

# Add content to hot memory
content = "My name is Adv, and I am a software engineer with 5 years of experience in Python and machine learning."
memory_manager.add_to_hot_memory(content)



