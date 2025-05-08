from interface.llm_interface import LLMInterface
from modules.entity_extractor import EntityExtractor
from modules.compressor import CompressorModule
from modules.relevance import RelevanceEvaluator
from core.memory_manager import MemoryManager
from modules.retriever import RetrievalEngine
from interface.integration import IntegrationLayer
from core.memory import MemorySegment, MemoryTier


class SelectiveCompressionSystem:


    """Main system class that integrates all components"""

    def init(self, model_name, hot_size=4000, warm_size=16000, cold_size=64000):
        # Initialize LLM interface
        self.llm_interface = LLMInterface(model_name)
            # Initialize entity extractor
        self.entity_extractor = EntityExtractor()
            
            # Initialize compressor
        self.compressor = CompressorModule(self.llm_interface, self.entity_extractor)
            
            # Initialize relevance evaluator
        self.relevance_evaluator = RelevanceEvaluator(self.llm_interface, self.entity_extractor)
            
            # Initialize memory manager
        self.memory_manager = MemoryManager(
                hot_size, warm_size, cold_size, 
                self.compressor, self.relevance_evaluator
            )
            
            # Initialize retrieval engine
        self.retrieval_engine = RetrievalEngine(
                self.memory_manager, self.relevance_evaluator, self.llm_interface
            )
            
            # Initialize integration layer
        self.integration_layer = IntegrationLayer(
                self.memory_manager, self.retrieval_engine, self.llm_interface
            )
            
    def process_input(self, user_input):
        """Process user input and generate response"""
        return self.integration_layer.process_user_input(user_input)
        
    def get_memory_stats(self):
        """Get statistics about memory usage"""
        return {
            'hot_memory': {
                'size': self.memory_manager.hot_memory.get_size(),
                'capacity': self.memory_manager.hot_memory.max_size,
                'segment_count': len(self.memory_manager.hot_memory.segments)
            },
            'warm_memory': {
                'size': self.memory_manager.warm_memory.get_size(),
                'capacity': self.memory_manager.warm_memory.max_size,
                'segment_count': len(self.memory_manager.warm_memory.segments)
            },
            'cold_memory': {
                'size': self.memory_manager.cold_memory.get_size(),
                'capacity': self.memory_manager.cold_memory.max_size,
                'segment_count': len(self.memory_manager.cold_memory.segments)
            }
        }