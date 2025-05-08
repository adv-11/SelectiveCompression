
import time

class MemorySegment:
    def __init__(self, content, id, creation_time, importance=0.5, compression_level=0):
        self.content = content
        self.id = id
        self.creation_time = creation_time
        self.last_accessed_time = creation_time
        self.access_count = 0
        self.importance_score = importance
        self.compression_level = compression_level
        self.metadata = {}
    
    def update_access(self):
        self.last_accessed_time = time.time()
        self.access_count += 1
        
    def get_size(self):
        # Estimate token count
        return len(self.content.split())



class MemoryTier:
    def __init__(self, name, max_size):
        self.name = name
        self.max_size = max_size
        self.segments = {}
        
    def add_segment(self, segment):
        self.segments[segment.id] = segment
        
    def remove_segment(self, segment_id):
        if segment_id in self.segments:
            del self.segments[segment_id]
            
    def get_size(self):
        return sum(segment.get_size() for segment in self.segments.values())
        
    def get_segment(self, segment_id):
        return self.segments.get(segment_id)
        
    def get_all_segments(self):
        return list(self.segments.values())