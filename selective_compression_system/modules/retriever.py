

class RetrievalEngine:
    def __init__(self, memory_manager, relevance_evaluator, llm_interface):
        self.memory_manager = memory_manager
        self.relevance_evaluator = relevance_evaluator
        self.llm_interface = llm_interface
        
    def identify_relevant_segments(self, current_context, user_query=None):
        relevant_segments = []
        
        # Check warm memory
        for segment in self.memory_manager.warm_memory.get_all_segments():
            relevance = self.relevance_evaluator.calculate_importance(
                segment, current_context, user_query
            )
            if relevance > 0.6:  
                relevant_segments.append((segment, relevance))
        
        # Check cold memory
        for segment in self.memory_manager.cold_memory.get_all_segments():
            relevance = self.relevance_evaluator.calculate_importance(
                segment, current_context, user_query
            )
            if relevance > 0.7:  # Higher threshold for cold memory
                relevant_segments.append((segment, relevance))
        
        # Sort by relevance
        relevant_segments.sort(key=lambda x: x[1], reverse=True)
        
        # Return top segments (respect memory budget)
        return [segment for segment, _ in relevant_segments[:10]]
    
    def decompress_segment(self, segment):
        # Generate appropriate prompt based on compression level
        if segment.compression_level == 1:
            prompt = self._generate_light_decompression_prompt(segment)
        else:
            prompt = self._generate_heavy_decompression_prompt(segment)
            
        # Get expanded content from LLM
        expanded_content = self.llm_interface.generate(prompt)
        
        # Create new segment with expanded content
        decompressed_segment = MemorySegment(
            content=expanded_content,
            id=segment.id,
            creation_time=segment.creation_time,
            importance=segment.importance_score,
            compression_level=0
        )
        
        # Preserve metadata and update access stats
        decompressed_segment.metadata = segment.metadata.copy() 
        decompressed_segment.metadata['decompression_time'] = time.time()
        decompressed_segment.last_accessed_time = time.time()
        decompressed_segment.access_count = segment.access_count + 1
        
        return decompressed_segment
        
    def _generate_light_decompression_prompt(self, segment):
        prompt = f"""
        Expand the following compressed conversation segment into a detailed form.
        Maintain all facts, entities, and relationships mentioned in the compressed version.
        
        Compressed segment:
        {segment.content}
        
        Detailed expansion:
        """
        return prompt
        
    def _generate_heavy_decompression_prompt(self, segment):
        entities = segment.metadata.get('original_entities', [])
        entity_str = ", ".join(entities) if entities else "Unknown"
        
        prompt = f"""
        Convert this heavily compressed information into a detailed, natural language form.
        The original text discussed these entities: {entity_str}
        
        Maintain all facts and relationships while expanding into natural language.
        
        Compressed information:
        {segment.content}
        
        Detailed expansion:
        """
        return prompt