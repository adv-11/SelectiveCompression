


class RelevanceEvaluator:
    def __init__(self, embedding_model, entity_extractor):
        self.embedding_model = embedding_model
        self.entity_extractor = entity_extractor
        
        # Constants for scoring
        self.RECENCY_DECAY_FACTOR = 0.01  # Controls time decay rate
        self.ACCESS_RECENCY_DECAY_FACTOR = 0.005  # Controls access time decay rate
        self.ACCESS_COUNT_FACTOR = 0.2  # Controls impact of access count
        self.ALPHA = 0.7  # Weight between recency and frequency in access score
        self.BETA = 0.5  # Weight between entity overlap count and importance
        
        # Scoring weights
        self.weights = {
            'recency': 0.20,
            'access': 0.15,
            'semantic': 0.30,
            'entity': 0.25,
            'query': 0.10
        }
        
    def calculate_importance(self, segment, current_context, user_query=None):
        # Calculate various factors
        recency_score = self._calculate_recency_score(segment)
        access_score = self._calculate_access_score(segment)
        semantic_score = self._calculate_semantic_similarity(segment, current_context)
        entity_score = self._calculate_entity_importance(segment, current_context)
        
        # Additional query relevance if available
        query_score = 0
        if user_query:
            query_score = self._calculate_query_relevance(segment, user_query)
            
        # Combine scores with weights
        importance_score = (
            self.weights['recency'] * recency_score +
            self.weights['access'] * access_score +
            self.weights['semantic'] * semantic_score +
            self.weights['entity'] * entity_score +
            self.weights['query'] * query_score
        )
        
        # Adjust for explicit importance markers
        if self._has_explicit_importance(segment):
            importance_score *= 1.5
        
        return importance_score
        
    def _calculate_recency_score(self, segment):

        """Calculate score based on how recent the segment is"""

        age = time.time() - segment.creation_time

        return math.exp(-self.RECENCY_DECAY_FACTOR * age)
        



    def _calculate_access_score(self, segment):

        """Calculate score based on access patterns"""

        recency_factor = math.exp(-self.ACCESS_RECENCY_DECAY_FACTOR * (time.time() - segment.last_accessed_time))

        frequency_factor = 1 - math.exp(-self.ACCESS_COUNT_FACTOR * segment.access_count)

        return self.ALPHA * recency_factor + (1 - self.ALPHA) * frequency_factor

    def _calculate_semantic_similarity(self, segment, current_context):

        """Calculate semantic similarity between segment and current context"""

        if not self.embedding_model:
            # Fallback to simple overlap if no embedding model
            return self._calculate_text_overlap(segment.content, current_context)
            
        # Get embeddings
        try:
            segment_embedding = self.embedding_model.embed_text(segment.content)
            context_embedding = self.embedding_model.embed_text(current_context)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(segment_embedding, context_embedding)
            return similarity
        except Exception:
            # Fallback if embedding fails
            return self._calculate_text_overlap(segment.content, current_context)

    def _calculate_entity_importance(self, segment, current_context):

        """Calculate importance based on entity overlap and importance"""

        # Extract entities
        segment_entities = self.entity_extractor.extract_entities(segment.content)
        context_entities = self.entity_extractor.extract_entities(current_context)
        
        if not segment_entities or not context_entities:
            return 0.0
            
        # Calculate overlap
        overlap = set(segment_entities).intersection(set(context_entities))
        overlap_ratio = len(overlap) / len(segment_entities)
        
        # Could be extended with global entity importance dictionary
        return overlap_ratio

    def _calculate_query_relevance(self, segment, user_query):

        """Calculate relevance to specific user query"""
        
        if not user_query:
            return 0.0
            
        # Simple approach: use semantic similarity
        if self.embedding_model:
            segment_embedding = self.embedding_model.embed_text(segment.content)
            query_embedding = self.embedding_model.embed_text(user_query)
            return self._cosine_similarity(segment_embedding, query_embedding)
        else:
            # Fallback to text overlap
            return self._calculate_text_overlap(segment.content, user_query)

    def _calculate_text_overlap(self, text1, text2):

        """Calculate simple word overlap between two texts"""

        # Simple tokenization by splitting on whitespace
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


    def _cosine_similarity(self, vec1, vec2):

        """Calculate cosine similarity between two vectors"""

        dot_product = sum(x * y for x, y in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(x * x for x in vec1))
        magnitude2 = math.sqrt(sum(y * y for y in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)



    def _has_explicit_importance(self, segment):

        """Check if segment has explicit importance markers"""

        importance_markers = [
            "important", "critical", "crucial", "key", "essential",
            "remember", "note", "significant", "vital", "remember this"
        ]
        
        for marker in importance_markers:
            if marker in segment.content.lower():
                return True
                
        return False