class MemoryManager:
    def __init__(self, hot_size, warm_size, cold_size, compressor, relevance_evaluator):
        self.hot_memory = MemoryTier("hot", hot_size)
        self.warm_memory = MemoryTier("warm", warm_size)
        self.cold_memory = MemoryTier("cold", cold_size)
        self.compressor = compressor
        self.relevance_evaluator = relevance_evaluator
        
    def add_to_hot_memory(self, content):
        segment = MemorySegment(
            content=content,
            id=str(uuid.uuid4()),
            creation_time=time.time()
        )
        self.hot_memory.add_segment(segment)
        
    def manage_memory_tiers(self):
        # Check and manage hot memory
        if self.hot_memory.get_size() > self.hot_memory.max_size * 0.9:
            self._compress_from_hot_to_warm()
            
        # Check and manage warm memory
        if self.warm_memory.get_size() > self.warm_memory.max_size * 0.9:
            self._compress_from_warm_to_cold()
            
        # Check and manage cold memory
        if self.cold_memory.get_size() > self.cold_memory.max_size * 0.9:
            self._evict_from_cold()
    
    def _compress_from_hot_to_warm(self):
        # Get current context from newest segments
        current_context = self._get_recent_context()
        
        # Score all segments in hot memory
        segments = self.hot_memory.get_all_segments()
        scored_segments = []
        
        for segment in segments:
            importance = self.relevance_evaluator.calculate_importance(
                segment, current_context
            )
            segment.importance_score = importance
            scored_segments.append((segment, importance))
            
        # Sort by importance (ascending - least important first)
        scored_segments.sort(key=lambda x: x[1])
        
        # Calculate how much we need to compress
        current_size = self.hot_memory.get_size()
        target_size = self.hot_memory.max_size * 0.7  # Reduce to 70%
        size_to_compress = current_size - target_size
        
        # Compress and move least important segments until we meet target
        compressed_size = 0
        for segment, _ in scored_segments:
            if compressed_size >= size_to_compress:
                break
                
            # Skip very recent segments (last 10 interactions)
            if len(segments) > 10 and segment in segments[-10:]:
                continue
                
            # Compress the segment
            compressed_segment = self.compressor.compress(segment, level=1)
            
            # Move to warm memory
            self.warm_memory.add_segment(compressed_segment)
            self.hot_memory.remove_segment(segment.id)
            
            # Update compressed size
            compressed_size += segment.get_size()
            
    def _compress_from_warm_to_cold(self):
        # Similar to hot to warm, but with different thresholds
        # Get all segments in warm memory
        segments = self.warm_memory.get_all_segments()
        current_context = self._get_recent_context()
        
        # Score all segments
        scored_segments = []
        for segment in segments:
            importance = self.relevance_evaluator.calculate_importance(
                segment, current_context
            )
            segment.importance_score = importance
            scored_segments.append((segment, importance))
            
        # Sort by importance (ascending)
        scored_segments.sort(key=lambda x: x[1])
        
        # Calculate compression target
        current_size = self.warm_memory.get_size()
        target_size = self.warm_memory.max_size * 0.7
        size_to_compress = current_size - target_size
        
        # Compress and move segments
        compressed_size = 0
        for segment, _ in scored_segments:
            if compressed_size >= size_to_compress:
                break
                
            # Compress the segment (level 2 - heavy compression)
            compressed_segment = self.compressor.compress(segment, level=2)
            
            # Move to cold memory
            self.cold_memory.add_segment(compressed_segment)
            self.warm_memory.remove_segment(segment.id)
            
            # Update compressed size
            compressed_size += segment.get_size()
    
    def _evict_from_cold(self):
        # Get all segments in cold memory
        segments = self.cold_memory.get_all_segments()
        current_context = self._get_recent_context()
        
        # Score all segments
        scored_segments = []
        for segment in segments:
            importance = self.relevance_evaluator.calculate_importance(
                segment, current_context
            )
            segment.importance_score = importance
            scored_segments.append((segment, importance))
            
        # Sort by importance (ascending)
        scored_segments.sort(key=lambda x: x[1])
        
        # Calculate eviction target
        current_size = self.cold_memory.get_size()
        target_size = self.cold_memory.max_size * 0.7
        size_to_evict = current_size - target_size
        
        # Evict segments
        evicted_size = 0
        for segment, _ in scored_segments:
            if evicted_size >= size_to_evict:
                break
                
            # Remove from cold memory
            self.cold_memory.remove_segment(segment.id)
            
            # Update evicted size
            evicted_size += segment.get_size()
            
    def _get_recent_context(self):
        # Get most recent segments from hot memory
        segments = self.hot_memory.get_all_segments()
        segments.sort(key=lambda x: x.creation_time, reverse=True)
        
        # Take most recent segments up to a certain size
        recent_segments = []
        recent_size = 0
        for segment in segments:
            if recent_size > 1000:  # Approximately 1000 tokens
                break
            recent_segments.append(segment)
            recent_size += segment.get_size()
            
        # Concatenate content
        return "\n".join([segment.content for segment in recent_segments])