import logging
import os
from typing import Dict, Any, Optional

from interface.llm_interface import LLMInterface
from modules.entity_extractor import EntityExtractor
from modules.compressor import CompressorModule
from modules.relevance import RelevanceEvaluator
from core.memory_manager import MemoryManager
from modules.retriever import RetrievalEngine
from interface.integration import IntegrationLayer
from core.memory import MemorySegment, MemoryTier


class SelectiveCompressionSystem:
    """Main system class that integrates all components of the selective compression memory system."""

    def __init__(self, model_name="gpt-4o-mini", hot_size=4000, warm_size=16000, cold_size=64000, api_key=None):
        """Initialize the selective compression system.
        
        Args:
            model_name (str): Name of the LLM model to use
            hot_size (int): Maximum size of hot memory tier in tokens
            warm_size (int): Maximum size of warm memory tier in tokens
            cold_size (int): Maximum size of cold memory tier in tokens
            api_key (str, optional): API key for LLM service
        """
        logging.info(f"Initializing SelectiveCompressionSystem with model {model_name}")
        
        # Initialize LLM interface
        self.llm_interface = LLMInterface(model_name, api_key)
        logging.info("Initialized LLM interface")
        
        # Initialize entity extractor
        self.entity_extractor = EntityExtractor()
        logging.info("Initialized entity extractor")
        
        # Initialize compressor
        self.compressor = CompressorModule(self.llm_interface, self.entity_extractor)
        logging.info("Initialized compressor module")
        
        # Initialize relevance evaluator
        self.relevance_evaluator = RelevanceEvaluator(self.llm_interface, self.entity_extractor)
        logging.info("Initialized relevance evaluator")
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            hot_size, warm_size, cold_size, 
            self.compressor, self.relevance_evaluator
        )
        logging.info("Initialized memory manager")
        
        # Initialize retrieval engine
        self.retrieval_engine = RetrievalEngine(
            self.memory_manager, self.relevance_evaluator, self.llm_interface
        )
        logging.info("Initialized retrieval engine")
        
        # Initialize integration layer
        self.integration_layer = IntegrationLayer(
            self.memory_manager, self.retrieval_engine, self.llm_interface
        )
        logging.info("Initialized integration layer")
        
        logging.info("Selective Compression System initialization complete")
    
    def process_input(self, user_input: str) -> str:
        """Process user input and generate response.
        
        Args:
            user_input (str): Input from the user
            
        Returns:
            str: Generated response
        """
        try:
            logging.info(f"Processing user input: {user_input[:50]}...")
            return self.integration_layer.process_user_input(user_input)
        except Exception as e:
            logging.error(f"Error in process_input: {str(e)}")
            return f"I encountered an error processing your input. Please try again."
    
    def get_memory_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about memory usage.
        
        Returns:
            Dict: Dictionary with memory statistics
        """
        try:
            return {
                'hot_memory': {
                    'size': self.memory_manager.hot_memory.get_size(),
                    'capacity': self.memory_manager.hot_memory.max_size,
                    'segment_count': len(self.memory_manager.hot_memory.segments),
                    'utilization': f"{(self.memory_manager.hot_memory.get_size() / self.memory_manager.hot_memory.max_size) * 100:.1f}%"
                },
                'warm_memory': {
                    'size': self.memory_manager.warm_memory.get_size(),
                    'capacity': self.memory_manager.warm_memory.max_size,
                    'segment_count': len(self.memory_manager.warm_memory.segments),
                    'utilization': f"{(self.memory_manager.warm_memory.get_size() / self.memory_manager.warm_memory.max_size) * 100:.1f}%"
                },
                'cold_memory': {
                    'size': self.memory_manager.cold_memory.get_size(),
                    'capacity': self.memory_manager.cold_memory.max_size,
                    'segment_count': len(self.memory_manager.cold_memory.segments),
                    'utilization': f"{(self.memory_manager.cold_memory.get_size() / self.memory_manager.cold_memory.max_size) * 100:.1f}%"
                },
                'total': {
                    'size': (self.memory_manager.hot_memory.get_size() + 
                             self.memory_manager.warm_memory.get_size() + 
                             self.memory_manager.cold_memory.get_size()),
                    'capacity': (self.memory_manager.hot_memory.max_size + 
                                self.memory_manager.warm_memory.max_size + 
                                self.memory_manager.cold_memory.max_size),
                    'segment_count': (len(self.memory_manager.hot_memory.segments) + 
                                     len(self.memory_manager.warm_memory.segments) + 
                                     len(self.memory_manager.cold_memory.segments))
                }
            }
        except Exception as e:
            logging.error(f"Error getting memory stats: {str(e)}")
            return {'error': str(e)}
            
    def reset_memory(self) -> None:
        """Reset all memory tiers."""
        try:
            self.memory_manager.hot_memory.segments = {}
            self.memory_manager.warm_memory.segments = {}
            self.memory_manager.cold_memory.segments = {}
            logging.info("Memory reset complete")
        except Exception as e:
            logging.error(f"Error resetting memory: {str(e)}")
            
    def get_hot_memory_contents(self) -> str:
        """Get the contents of hot memory as a string.
        
        Returns:
            str: Concatenated content of hot memory segments
        """
        try:
            hot_segments = self.memory_manager.hot_memory.get_all_segments()
            
            # Sort by creation time
            hot_segments.sort(key=lambda x: x.creation_time)
            
            # Concatenate content
            if hot_segments:
                return "\n\n---\n\n".join([segment.content for segment in hot_segments])
            else:
                return "Hot memory is empty"
        except Exception as e:
            logging.error(f"Error getting hot memory contents: {str(e)}")
            return f"Error: {str(e)}"