from core.memory import MemorySegment
import time

class CompressorModule:
    def __init__(self, llm_interface, entity_extractor):
        self.llm_interface = llm_interface
        self.entity_extractor = entity_extractor
        
    def compress(self, segment, level):
        if level == 1:  # Light compression
            return self._light_compression(segment)
        elif level == 2:  # Heavy compression
            return self._heavy_compression(segment)
            
    def _light_compression(self, segment):
        # Extract entities for reference
        entities = self.entity_extractor.extract_entities(segment.content)
        
        # Generate prompt for light compression
        prompt = self._generate_light_compression_prompt(segment.content, entities)
        
        # Get compression from LLM
        compressed_content = self.llm_interface.generate(prompt)
        
        # Create new segment with compressed content
        compressed_segment = MemorySegment(
            content=compressed_content,
            id=segment.id,
            creation_time=segment.creation_time,
            importance=segment.importance_score,
            compression_level=1
        )
        
        # Copy and update metadata
        compressed_segment.metadata = segment.metadata.copy() if hasattr(segment, 'metadata') else {}
        compressed_segment.metadata.update({
            'original_length': len(segment.content),
            'compressed_length': len(compressed_content),
            'compression_ratio': len(segment.content) / max(1, len(compressed_content)),
            'original_entities': entities,
            'compression_timestamp': time.time(),
            'compression_level': 1
        })
        
        # Copy access information
        compressed_segment.last_accessed_time = segment.last_accessed_time
        compressed_segment.access_count = segment.access_count
        
        return compressed_segment
    
    def _heavy_compression(self, segment):
        # Extract entities and relationships
        entities = self.entity_extractor.extract_entities(segment.content)
        relationships = self.entity_extractor.extract_relationships(segment.content, entities)
        
        # Generate prompt for heavy compression
        prompt = self._generate_heavy_compression_prompt(segment.content, entities, relationships)
        
        # Get compression from LLM
        compressed_content = self.llm_interface.generate(prompt)
        
        # Create new segment with heavily compressed content
        compressed_segment = MemorySegment(
            content=compressed_content,
            id=segment.id,
            creation_time=segment.creation_time,
            importance=segment.importance_score,
            compression_level=2
        )
        
        # Copy and update metadata
        compressed_segment.metadata = segment.metadata.copy() if hasattr(segment, 'metadata') else {}
        compressed_segment.metadata.update({
            'original_length': len(segment.content),
            'compressed_length': len(compressed_content),
            'compression_ratio': len(segment.content) / max(1, len(compressed_content)),
            'original_entities': entities,
            'original_relationships': relationships,
            'compression_timestamp': time.time(),
            'compression_level': 2
        })
        
        # Copy access information
        compressed_segment.last_accessed_time = segment.last_accessed_time
        compressed_segment.access_count = segment.access_count
        
        return compressed_segment
        
    def _generate_light_compression_prompt(self, content, entities):
        entity_str = ", ".join(entities) if entities else "No significant entities detected"
        
        prompt = f"""
        Compress the following conversation segment into a shorter form while preserving the key information.
        Keep all important facts, opinions, and context.
        
        Important entities to preserve: {entity_str}
        
        Original content:
        {content}
        
        Compressed version:
        """
        return prompt
        
    def _generate_heavy_compression_prompt(self, content, entities, relationships):
        entity_str = ", ".join(entities) if entities else "No significant entities detected"
        relationship_str = "; ".join([f"{r[0]} - {r[1]} - {r[2]}" for r in relationships]) if relationships else "No significant relationships detected"
        
        prompt = f"""
        Create an extremely compressed representation of the following content.
        Focus only on capturing the essential meaning, key entities, and their relationships.
        
        Important entities: {entity_str}
        Important relationships: {relationship_str}
        
        Original content:
        {content}
        
        Highly compressed representation:
        """
        return prompt