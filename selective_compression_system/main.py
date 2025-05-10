#!/usr/bin/env python
import os
import sys
import time
import logging
import argparse
from typing import Dict, Any, Optional
from core.memory import MemorySegment

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.system import SelectiveCompressionSystem
from modules.entity_extractor import EntityExtractor
from modules.compressor import CompressorModule
from interface.llm_interface import LLMInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('selective_compression_debug.log')
    ]
)


class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing without API calls"""
    
    def __init__(self, model_name, api_key=None):
        """Initialize with default values but don't connect to OpenAI"""
        self.model_name = model_name
        self.api_key = api_key
        logging.info(f"Initialized MockLLMInterface with model {model_name}")
    
    def generate(self, prompt: str) -> str:
        """Mock text generation"""
        logging.info(f"Mock generate called with prompt length: {len(prompt)}")
        
        # Generate mock response based on the prompt
        if "compress" in prompt.lower():
            return f"COMPRESSED: {prompt[-100:]}..."
        elif "expand" in prompt.lower() or "decompress" in prompt.lower():
            return f"EXPANDED: This is a detailed expansion of the compressed information. {prompt[-100:]}..."
        else:
            return f"RESPONSE: I'm a mock LLM response to: {prompt[:50]}..."
    
    def generate_response(self, context: str, user_input: str) -> str:
        """Generate a mock response to user input"""
        logging.info(f"Mock generate_response called with context length: {len(context)} and user input: {user_input[:50]}")
        return f"I'm responding to '{user_input[:30]}...' based on the provided context."
    
    def embed_text(self, text: str) -> list:
        """Generate mock embeddings"""
        import random
        logging.info(f"Mock embed_text called with text length: {len(text)}")
        # Return random vector with consistent dimensions for same input
        random.seed(hash(text) % 10000)
        return [random.random() for _ in range(1536)]


def test_memory_tiers():
    """Test basic memory tier functionality"""
    logging.info("Testing memory tiers...")
    
    from core.memory import MemorySegment, MemoryTier
    
    # Create memory tiers
    hot_tier = MemoryTier("hot", 1000)
    
    # Create test segments
    segment1 = MemorySegment("This is test content 1", "1", time.time())
    segment2 = MemorySegment("This is test content 2", "2", time.time())
    
    # Add segments to tier
    hot_tier.add_segment(segment1)
    hot_tier.add_segment(segment2)
    
    # Test retrieval
    assert hot_tier.get_segment("1") == segment1
    assert hot_tier.get_segment("2") == segment2
    
    # Test removal
    hot_tier.remove_segment("1")
    assert hot_tier.get_segment("1") is None
    assert len(hot_tier.get_all_segments()) == 1
    
    logging.info("Memory tier tests passed!")
    return True


def test_entity_extraction():
    """Test entity extraction capabilities"""
    logging.info("Testing entity extraction...")
    
    # Create entity extractor
    entity_extractor = EntityExtractor()
    
    # Test text
    test_text = """
    John Smith met with Sarah Johnson at Google headquarters in Mountain View. 
    They discussed the new AI project and its impact on the technology industry.
    Apple and Microsoft are also working on similar initiatives.
    """
    
    # Extract entities
    entities = entity_extractor.extract_entities(test_text)
    logging.info(f"Extracted entities: {entities}")
    
    # Extract relationships
    relationships = entity_extractor.extract_relationships(test_text, entities)
    logging.info(f"Extracted relationships: {relationships}")
    
    # Verify some entities were extracted
    assert len(entities) > 0
    
    logging.info("Entity extraction tests passed!")
    return True


def test_compression_decompression():
    """Test compression and decompression functionality"""
    logging.info("Testing compression and decompression...")
    
    # Initialize mock LLM interface
    llm = MockLLMInterface("gpt-4o-mini")
    
    # Create entity extractor and compressor
    entity_extractor = EntityExtractor()
    compressor = CompressorModule(llm, entity_extractor)
    
    # Create test segment
    test_segment = MemorySegment(
        content="This is a long conversation about artificial intelligence and its impact on society. Many experts believe AI will transform industries like healthcare, education, and transportation. However, there are also concerns about privacy, security, and job displacement.",
        id="test1",
        creation_time=time.time()
    )
    
    # Test light compression
    compressed_segment = compressor.compress(test_segment, level=1)
    logging.info(f"Light compression result: {compressed_segment.content}")
    
    # Test heavy compression
    heavy_compressed = compressor.compress(test_segment, level=2)
    logging.info(f"Heavy compression result: {heavy_compressed.content}")
    
    # Check compression metadata
    assert compressed_segment.compression_level == 1
    assert "compression_ratio" in compressed_segment.metadata
    assert heavy_compressed.compression_level == 2
    
    # Test decompression
    from modules.retriever import RetrievalEngine
    from modules.relevance import RelevanceEvaluator
    from core.memory_manager import MemoryManager
    
    relevance_evaluator = RelevanceEvaluator(llm, entity_extractor)
    memory_manager = MemoryManager(1000, 4000, 16000, compressor, relevance_evaluator)
    retrieval_engine = RetrievalEngine(memory_manager, relevance_evaluator, llm)
    
    # Decompress segments
    decompressed_light = retrieval_engine.decompress_segment(compressed_segment)
    decompressed_heavy = retrieval_engine.decompress_segment(heavy_compressed)
    
    logging.info(f"Light decompression result: {decompressed_light.content}")
    logging.info(f"Heavy decompression result: {decompressed_heavy.content}")
    
    # Check decompression results
    assert decompressed_light.compression_level == 0
    assert decompressed_heavy.compression_level == 0
    assert "decompression_time" in decompressed_light.metadata
    
    logging.info("Compression and decompression tests passed!")
    return True


def test_memory_management():
    """Test memory management functionality"""
    logging.info("Testing memory management...")
    
    # Initialize mock LLM interface
    llm = MockLLMInterface("gpt-4o-mini")
    
    # Create system with small memory sizes to force management operations
    system = SelectiveCompressionSystem(
        model_name="mock-model",
        hot_size=200,   # Small size to force compression
        warm_size=400,  # Small size to force compression
        cold_size=800   # Small size to force eviction
    )
    
    # Replace LLM with mock
    system.llm_interface = llm
    
    # Add several entries to hot memory to trigger management
    for i in range(20):
        content = f"Test message {i}: This is a test message with some content to make it longer than just a few words. We need to make sure it has sufficient size to trigger memory management operations when we add enough of these messages."
        system.memory_manager.add_to_hot_memory(content)
        
        # Log memory stats occasionally
        if i % 5 == 0:
            stats = system.get_memory_stats()
            logging.info(f"Memory stats after {i} additions: {stats}")
    
    # Force memory management
    result = system.force_memory_management()
    logging.info(f"Memory management result: {result}")
    
    # Verify tiers have content
    stats = system.get_memory_stats()
    assert stats["hot_memory"]["segment_count"] > 0
    
    # If compression worked, we should see content in warm memory
    assert stats["warm_memory"]["segment_count"] > 0
    
    logging.info("Memory management tests passed!")
    return True


def test_user_interaction():
    """Test user interaction flow"""
    logging.info("Testing user interaction flow...")
    
    # Initialize the system with mock LLM
    system = SelectiveCompressionSystem(model_name="mock-model")
    system.llm_interface = MockLLMInterface("gpt-4o-mini")
    
    # Simulate a conversation
    responses = []
    
    # First message
    response1 = system.process_input("Hello, my name is Alice.")
    responses.append(response1)
    logging.info(f"Response 1: {response1}")
    
    # Second message
    response2 = system.process_input("I'm interested in learning about selective compression systems.")
    responses.append(response2)
    logging.info(f"Response 2: {response2}")
    
    # Third message that should trigger some memory management
    response3 = system.process_input("Can you remember what my name was?")
    responses.append(response3)
    logging.info(f"Response 3: {response3}")
    
    # Check memory contents
    hot_memory = system.get_hot_memory_contents()
    logging.info(f"Hot memory contents: {hot_memory}")
    
    # Get memory stats
    stats = system.get_memory_stats()
    logging.info(f"Final memory stats: {stats}")
    
    # Verify we got responses
    assert all(responses)
    assert "Alice" in hot_memory
    
    logging.info("User interaction flow tests passed!")
    return True


def main():
    """Main function to run tests and demo the system"""
    parser = argparse.ArgumentParser(description="Selective Compression System Tester")
    parser.add_argument("--test", choices=["all", "memory", "entity", "compression", "management", "interaction"], 
                        default="all", help="Which test to run")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--openai-key", help="OpenAI API key (if not using mock)")
    parser.add_argument("--use-real-llm", action="store_true", help="Use real LLM instead of mock")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    
    # Run selected tests
    if args.test == "memory" or args.test == "all":
        test_memory_tiers()
    
    if args.test == "entity" or args.test == "all":
        test_entity_extraction()
    
    if args.test == "compression" or args.test == "all":
        test_compression_decompression()
    
    if args.test == "management" or args.test == "all":
        test_memory_management()
    
    if args.test == "interaction" or args.test == "all":
        test_user_interaction()
    
    # Run interactive demo if requested
    if args.demo:
        run_interactive_demo(use_real_llm=args.use_real_llm)


def run_interactive_demo(use_real_llm=False):
    """Run an interactive demo of the system"""
    print("\n" + "="*50)
    print("SELECTIVE COMPRESSION SYSTEM INTERACTIVE DEMO")
    print("="*50)
    
    # Initialize the system
    print("\nInitializing system...")
    
    if use_real_llm:
        if "OPENAI_API_KEY" not in os.environ:
            print("Error: OPENAI_API_KEY environment variable is required for real LLM usage.")
            print("Please set it or use --openai-key argument.")
            return
        
        system = SelectiveCompressionSystem(
            model_name="gpt-4o-mini",
            hot_size=4000,
            warm_size=16000,
            cold_size=64000
        )
        print("Using real OpenAI LLM")
    else:
        system = SelectiveCompressionSystem(
            model_name="mock-model",
            hot_size=4000,
            warm_size=16000,
            cold_size=64000
        )
        system.llm_interface = MockLLMInterface("gpt-4o-mini")
        print("Using mock LLM (responses will be generic)")
    
    print("\nSystem initialized! Type 'exit' to quit, 'stats' for memory stats, or 'memory' to view hot memory.\n")
    
    # Main interaction loop
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                print("Exiting demo. Goodbye!")
                break
            
            elif user_input.lower() == 'stats':
                stats = system.get_memory_stats()
                print("\nMEMORY STATISTICS:")
                print(f"HOT:  {stats['hot_memory']['segment_count']} segments, {stats['hot_memory']['size']} tokens ({stats['hot_memory']['utilization']})")
                print(f"WARM: {stats['warm_memory']['segment_count']} segments, {stats['warm_memory']['size']} tokens ({stats['warm_memory']['utilization']})")
                print(f"COLD: {stats['cold_memory']['segment_count']} segments, {stats['cold_memory']['size']} tokens ({stats['cold_memory']['utilization']})")
                print(f"TOTAL: {stats['total']['segment_count']} segments, {stats['total']['size']} tokens")
                continue
            
            elif user_input.lower() == 'memory':
                hot_memory = system.get_hot_memory_contents()
                print("\nHOT MEMORY CONTENTS:")
                print(hot_memory)
                continue
            
            elif user_input.lower() == 'manage':
                print("\nForcing memory management...")
                result = system.force_memory_management()
                print(f"Memory after management: Hot: {result['after']['hot_memory']['segment_count']} segments, Warm: {result['after']['warm_memory']['segment_count']} segments, Cold: {result['after']['cold_memory']['segment_count']} segments")
                continue
            
            elif user_input.lower() == 'reset':
                print("\nResetting memory...")
                system.reset_memory()
                print("Memory reset complete.")
                continue
            
            # Process normal input
            start_time = time.time()
            response = system.process_input(user_input)
            end_time = time.time()
            
            print(f"\nSystem: {response}")
            print(f"(Response time: {end_time - start_time:.2f}s)")
            
        except KeyboardInterrupt:
            print("\nExiting demo. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unhandled exception in main: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")