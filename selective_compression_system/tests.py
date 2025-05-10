#!/usr/bin/env python
import unittest
import time
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory import MemorySegment, MemoryTier
from modules.entity_extractor import EntityExtractor
from modules.compressor import CompressorModule
from core.system import SelectiveCompressionSystem
from modules.relevance import RelevanceEvaluator
from core.memory_manager import MemoryManager


# Mock LLM Interface for testing
class MockLLMInterface:
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.api_key = api_key
    
    def generate(self, prompt):
        return f"Mock response to: {prompt[:50]}..."
    
    def generate_response(self, context, user_input):
        return f"Mock response to: {user_input[:50]}..."
    
    def embed_text(self, text):
        # Return mock embeddings with consistent dimensions
        import random
        random.seed(hash(text) % 10000)
        return [random.random() for _ in range(1536)]


class TestMemoryModule(unittest.TestCase):
    def test_memory_segment(self):
        # Create memory segment
        segment = MemorySegment("Test content", "test-id", time.time())
        
        # Check attributes
        self.assertEqual(segment.content, "Test content")
        self.assertEqual(segment.id, "test-id")
        self.assertEqual(segment.compression_level, 0)
        
        # Check size estimation
        self.assertEqual(segment.get_size(), 2)  # "Test content" has 2 words
        
        # Test access update
        old_access_time = segment.last_accessed_time
        old_access_count = segment.access_count
        segment.update_access()
        self.assertGreater(segment.last_accessed_time, old_access_time)
        self.assertEqual(segment.access_count, old_access_count + 1)
    
    def test_memory_tier(self):
        # Create memory tier
        tier = MemoryTier("test-tier", 1000)
        
        # Create segments
        segment1 = MemorySegment("Content 1", "id1", time.time())
        segment2 = MemorySegment("Content 2", "id2", time.time())
        
        # Add segments
        tier.add_segment(segment1)
        tier.add_segment(segment2)
        
        # Check segment retrieval
        self.assertEqual(tier.get_segment("id1"), segment1)
        self.assertEqual(tier.get_segment("id2"), segment2)
        self.assertIsNone(tier.get_segment("non-existent"))
        
        # Check segment listing
        all_segments = tier.get_all_segments()
        self.assertEqual(len(all_segments), 2)
        self.assertIn(segment1, all_segments)
        self.assertIn(segment2, all_segments)
        
        # Check size calculation
        self.assertEqual(tier.get_size(), 4)  # "Content 1" + "Content 2" = 4 words
        
        # Remove segment
        tier.remove_segment("id1")
        self.assertIsNone(tier.get_segment("id1"))
        self.assertEqual(len(tier.get_all_segments()), 1)


class TestEntityExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = EntityExtractor()
    
    def test_entity_extraction(self):
        text = "Apple CEO Tim Cook announced the new iPhone in Cupertino, California."
        entities = self.extractor.extract_entities(text)
        
        # Check some expected entities are extracted
        self.assertGreater(len(entities), 0)
        
        # These might be extracted depending on the extraction method
        possible_entities = ["Apple", "Tim Cook", "iPhone", "Cupertino", "California"]
        found = False
        for entity in possible_entities:
            if any(entity.lower() in e.lower() for e in entities):
                found = True
                break
        
        self.assertTrue(found, f"None of {possible_entities} found in {entities}")
    
    def test_relationship_extraction(self):
        text = "Apple CEO Tim Cook announced the new iPhone in Cupertino, California."
        entities = self.extractor.extract_entities(text)
        relationships = self.extractor.extract_relationships(text, entities)
        
        # There should be some relationships
        self.assertIsInstance(relationships, list)


class TestCompressor(unittest.TestCase):
    def setUp(self):
        self.llm = MockLLMInterface("test-model")
        self.entity_extractor = EntityExtractor()
        self.compressor = CompressorModule(self.llm, self.entity_extractor)
    
    def test_light_compression(self):
        segment = MemorySegment("This is a test content that should be compressed. It contains some information about compression algorithms and memory management.", "test-id", time.time())
        
        compressed = self.compressor.compress(segment, level=1)
        
        # Check that compression happened
        self.assertEqual(compressed.compression_level, 1)
        self.assertEqual(compressed.id, segment.id)
        self.assertIsNotNone(compressed.metadata.get('compression_ratio'))
        self.assertIn('original_entities', compressed.metadata)
    
    def test_heavy_compression(self):
        segment = MemorySegment("This is a test content that should be heavily compressed. It contains information about compression algorithms and memory management that should be reduced to its core essence.", "test-id", time.time())
        
        compressed = self.compressor.compress(segment, level=2)
        
        # Check that compression happened
        self.assertEqual(compressed.compression_level, 2)
        self.assertEqual(compressed.id, segment.id)
        self.assertIsNotNone(compressed.metadata.get('compression_ratio'))
        self.assertIn('original_entities', compressed.metadata)
        self.assertIn('original_relationships', compressed.metadata)


class TestRelevanceEvaluator(unittest.TestCase):
    def setUp(self):
        self.llm = MockLLMInterface("test-model")
        self.entity_extractor = EntityExtractor()
        self.relevance_evaluator = RelevanceEvaluator(self.llm, self.entity_extractor)
    
    def test_relevance_calculation(self):
        segment = MemorySegment("This content is about artificial intelligence and machine learning.", "test-id", time.time())
        context = "Let's discuss artificial intelligence applications."
        
        # Calculate importance
        importance = self.relevance_evaluator.calculate_importance(segment, context)
        
        # Should return a float between 0 and 1
        self.assertIsInstance(importance, float)
        self.assertGreaterEqual(importance, 0.0)
        
        # Test with unrelated context
        unrelated_context = "The weather is nice today in California."
        unrelated_importance = self.relevance_evaluator.calculate_importance(segment, unrelated_context)
        
        # Related context should have higher importance than unrelated
        self.assertGreaterEqual(importance, unrelated_importance)


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.llm = MockLLMInterface("test-model")
        self.entity_extractor = EntityExtractor()
        self.compressor = CompressorModule(self.llm, self.entity_extractor)
        self.relevance_evaluator = RelevanceEvaluator(self.llm, self.entity_extractor)
        self.memory_manager = MemoryManager(
            100,  # small hot size for testing
            200,  # small warm size
            300,  # small cold size
            self.compressor,
            self.relevance_evaluator
        )
    
    def test_add_to_hot_memory(self):
        # Add content to hot memory
        self.memory_manager.add_to_hot_memory("This is test content for hot memory.")
        
        # Check that content was added
        self.assertEqual(len(self.memory_manager.hot_memory.segments), 1)
    
    def test_memory_management(self):
        # Add multiple segments to trigger management
        for i in range(20):
            self.memory_manager.add_to_hot_memory(f"This is test content {i} with sufficient length to take up space in the memory system.")
        
        # Force memory management
        self.memory_manager.manage_memory_tiers()
        
        # Check that segments were moved to warm memory
        self.assertGreater(len(self.memory_manager.warm_memory.segments), 0)


class TestFullSystem(unittest.TestCase):
    def setUp(self):
        # Create system with mock LLM
        self.system = SelectiveCompressionSystem(
            model_name="test-model",
            hot_size=100,  # small sizes for testing
            warm_size=200,
            cold_size=300
        )
        self.system.llm_interface = MockLLMInterface("test-model")
    
    def test_system_initialization(self):
        # Check system components initialized
        self.assertIsNotNone(self.system.memory_manager)
        self.assertIsNotNone(self.system.entity_extractor)
        self.assertIsNotNone(self.system.compressor)
        self.assertIsNotNone(self.system.relevance_evaluator)
        self.assertIsNotNone(self.system.retrieval_engine)
        self.assertIsNotNone(self.system.integration_layer)
    
    def test_process_input(self):
        # Process user input
        response = self.system.process_input("Hello, this is a test input")
        
        # Should get a response
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        
        # Check that input was stored in hot memory
        self.assertEqual(len(self.system.memory_manager.hot_memory.segments), 2)  # user input + response
    
    def test_memory_stats(self):
        # Add some content
        self.system.process_input("Hello, system")
        
        # Get memory stats
        stats = self.system.get_memory_stats()
        
        # Check structure
        self.assertIn('hot_memory', stats)
        self.assertIn('warm_memory', stats)
        self.assertIn('cold_memory', stats)
        self.assertIn('total', stats)
        
        # Check hot memory has content
        self.assertGreater(stats['hot_memory']['segment_count'], 0)
    
    def test_reset_memory(self):
        # Add some content
        self.system.process_input("Hello, system")
        
        # Reset memory
        self.system.reset_memory()
        
        # Check memory is empty
        stats = self.system.get_memory_stats()
        self.assertEqual(stats['hot_memory']['segment_count'], 0)
        self.assertEqual(stats['warm_memory']['segment_count'], 0)
        self.assertEqual(stats['cold_memory']['segment_count'], 0)


if __name__ == "__main__":
    unittest.main()