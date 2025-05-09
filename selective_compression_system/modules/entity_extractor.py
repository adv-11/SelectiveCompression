import re
import logging
from collections import Counter
from typing import List, Tuple, Dict, Set, Optional
import json

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("Spacy not available. Falling back to simple entity extraction.")

class EntityExtractor:
    def __init__(self, nlp_model=None):
        """Initialize entity extractor with optional NLP model.
        
        Args:
            nlp_model: Optional pre-loaded spaCy model
        """
        self.nlp_model = nlp_model
        
        # Try to load spaCy model if available and not provided
        if SPACY_AVAILABLE and not self.nlp_model:
            try:
                import spacy
                self.nlp_model = spacy.load("en_core_web_sm")
                logging.info("Loaded spaCy en_core_web_sm model")
            except Exception as e:
                logging.warning(f"Could not load spaCy model: {str(e)}")
        
        # Common English stopwords to filter out
        self.stopwords = set([
            "the", "and", "is", "of", "to", "a", "in", "that", "it", "with",
            "as", "for", "on", "was", "be", "this", "have", "are", "an", "by",
            "not", "but", "or", "at", "from", "they", "you", "he", "she", "his",
            "her", "we", "which", "what", "there", "when", "who", "how", "all",
            "been", "has", "their", "one", "would", "will", "can", "if", "more",
            "about", "up", "so", "no", "out", "them", "my", "your", "could", "than",
            "only", "should", "very", "some", "other", "into", "just", "over", "also"
        ])
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of extracted entities
        """
        if not text:
            return []
            
        if self.nlp_model and SPACY_AVAILABLE:
            return self._extract_with_model(text)
        else:
            return self._simple_entity_extraction(text)
            
    def extract_relationships(self, text: str, entities: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities.
        
        Args:
            text (str): Text to analyze
            entities (List[str], optional): Pre-extracted entities
            
        Returns:
            List[Tuple[str, str, str]]: List of (entity1, relationship, entity2) tuples
        """
        if not text:
            return []
            
        if not entities:
            entities = self.extract_entities(text)
            
        if self.nlp_model and SPACY_AVAILABLE:
            return self._extract_relationships_with_model(text, entities)
        else:
            return self._simple_relationship_extraction(text, entities)
            
    def _simple_entity_extraction(self, text: str) -> List[str]:
        """A word frequency and position-based entity extraction.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of extracted entities
        """
        # Clean and tokenize text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Remove stopwords and single characters
        filtered_words = [word for word in words if word not in self.stopwords and len(word) > 1]
        
        # Count word frequency
        word_counts = Counter(filtered_words)
        
        # Extract possible multi-word entities (bigrams)
        bigrams = [' '.join(filtered_words[i:i+2]) for i in range(len(filtered_words)-1)]
        bigram_counts = Counter(bigrams)
        
        # Combine single words and bigrams, giving more weight to words at the beginning
        entities = []
        position_weight = 1.5  # Words near the beginning get higher scores
        
        # Score single words with position weighting
        word_scores = {}
        for i, word in enumerate(filtered_words):
            if word not in self.stopwords and len(word) > 1:
                position_score = max(0, position_weight - (i / (len(filtered_words) + 1)))
                word_scores[word] = word_counts[word] + position_score
        
        # Add top scoring words
        entities.extend([word for word, _ in sorted(word_scores.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True)[:15]])
        
        # Add top bigrams that aren't already covered
        for bigram, count in bigram_counts.most_common(10):
            parts = bigram.split()
            if count > 1 and not all(part in entities for part in parts):
                entities.append(bigram)
        
        # Return unique entities
        return list(dict.fromkeys(entities))[:15]  # Remove duplicates while preserving order
        
    def _simple_relationship_extraction(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Extract relationships based on co-occurrence and simple patterns.
        
        Args:
            text (str): Text to analyze
            entities (List[str]): List of extracted entities
            
        Returns:
            List[Tuple[str, str, str]]: List of (entity1, relationship, entity2) tuples
        """
        relationships = []
        sentences = re.split(r'[.!?]', text)
        
        # Common relationship verbs
        relationship_verbs = [
            "is", "was", "are", "were", "has", "have", "had", "contains", "includes",
            "involves", "relates to", "belongs to", "owns", "created", "made",
            "developed", "leads", "causes", "affects", "influences", "depends on"
        ]
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if not sentence:
                continue
                
            # Find entities in this sentence
            present_entities = [entity for entity in entities 
                               if entity.lower() in sentence.lower()]
            
            # Basic co-occurrence relationship
            if len(present_entities) >= 2:
                for i in range(len(present_entities)):
                    for j in range(i+1, len(present_entities)):
                        # Look for relationship verbs between entities
                        e1_pos = sentence.find(present_entities[i])
                        e2_pos = sentence.find(present_entities[j])
                        
                        if e1_pos != -1 and e2_pos != -1:
                            if e1_pos < e2_pos:
                                between_text = sentence[e1_pos + len(present_entities[i]):e2_pos].strip()
                                for verb in relationship_verbs:
                                    if verb in between_text:
                                        relationships.append((present_entities[i], verb, present_entities[j]))
                                        break
                                else:  # No specific verb found
                                    relationships.append((present_entities[i], "related-to", present_entities[j]))
                            else:
                                between_text = sentence[e2_pos + len(present_entities[j]):e1_pos].strip()
                                for verb in relationship_verbs:
                                    if verb in between_text:
                                        relationships.append((present_entities[j], verb, present_entities[i]))
                                        break
                                else:  # No specific verb found
                                    relationships.append((present_entities[j], "related-to", present_entities[i]))
        
        # Remove duplicates while preserving order
        unique_relationships = []
        seen = set()
        for rel in relationships:
            rel_tuple = (rel[0], rel[1], rel[2])
            if rel_tuple not in seen:
                seen.add(rel_tuple)
                unique_relationships.append(rel)
        
        return unique_relationships[:10]  # Limit to top 10 relationships
        
    def _extract_with_model(self, text: str) -> List[str]:
        """Use spaCy NLP model for entity extraction.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of extracted entities
        """
        try:
            # Process text with spaCy
            doc = self.nlp_model(text)
            
            # Extract named entities
            named_entities = []
            for ent in doc.ents:
                # Filter out date and numerical entities
                if ent.label_ not in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                    named_entities.append(ent.text)
            
            # Extract noun chunks as potential entities
            noun_chunks = [chunk.text for chunk in doc.noun_chunks 
                          if len(chunk.text.split()) <= 3]  # Limit to 3 words
            
            # Combine and prioritize named entities
            all_entities = named_entities + [nc for nc in noun_chunks 
                                           if nc not in named_entities]
            
            # Remove duplicates while preserving order
            unique_entities = []
            seen = set()
            for entity in all_entities:
                if entity.lower() not in seen and entity.lower() not in self.stopwords:
                    seen.add(entity.lower())
                    unique_entities.append(entity)
            
            return unique_entities[:15]  # Limit to top 15 entities
            
        except Exception as e:
            logging.error(f"Error in spaCy entity extraction: {str(e)}")
            # Fallback to simple extraction
            return self._simple_entity_extraction(text)
        
    def _extract_relationships_with_model(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Use spaCy NLP model for relationship extraction.
        
        Args:
            text (str): Text to analyze
            entities (List[str]): List of extracted entities
            
        Returns:
            List[Tuple[str, str, str]]: List of (entity1, relationship, entity2) tuples
        """
        try:
            # Process text with spaCy
            doc = self.nlp_model(text)
            
            relationships = []
            entity_spans = {}
            
            # Find all entity spans in the text
            for entity in entities:
                entity_lower = entity.lower()
                text_lower = text.lower()
                
                start = 0
                while start < len(text_lower):
                    pos = text_lower.find(entity_lower, start)
                    if pos == -1:
                        break
                    
                    entity_spans[pos] = (entity, pos, pos + len(entity))
                    start = pos + 1
            
            # For each sentence, extract relationships between entities
            for sent in doc.sents:
                sent_text = sent.text
                sent_start = sent.start_char
                
                # Find entities in this sentence
                sent_entities = []
                for start_pos, (entity, abs_start, abs_end) in entity_spans.items():
                    if sent_start <= abs_start and abs_end <= sent_start + len(sent_text):
                        sent_entities.append((entity, abs_start - sent_start, abs_end - sent_start))
                
                # Extract relationships between pairs of entities
                for i, (entity1, start1, end1) in enumerate(sent_entities):
                    for j, (entity2, start2, end2) in enumerate(sent_entities[i+1:], i+1):
                        # Skip self-relationships
                        if entity1 == entity2:
                            continue
                            
                        # Extract relationship based on dependency parsing
                        if start1 < start2:
                            between_span = sent_text[end1:start2].strip()
                            
                            # Extract verb or prep from between text
                            between_doc = self.nlp_model(between_span)
                            verbs = [token.text for token in between_doc if token.pos_ == "VERB"]
                            preps = [token.text for token in between_doc if token.pos_ == "ADP"]
                            
                            relation = "related-to"
                            if verbs:
                                relation = verbs[0]
                            elif preps:
                                relation = preps[0]
                                
                            relationships.append((entity1, relation, entity2))
                        else:
                            between_span = sent_text[end2:start1].strip()
                            
                            # Extract verb or prep from between text
                            between_doc = self.nlp_model(between_span)
                            verbs = [token.text for token in between_doc if token.pos_ == "VERB"]
                            preps = [token.text for token in between_doc if token.pos_ == "ADP"]
                            
                            relation = "related-to"
                            if verbs:
                                relation = verbs[0]
                            elif preps:
                                relation = preps[0]
                                
                            relationships.append((entity2, relation, entity1))
            
            # Remove duplicates while preserving order
            unique_relationships = []
            seen = set()
            for rel in relationships:
                rel_tuple = (rel[0], rel[1], rel[2])
                if rel_tuple not in seen:
                    seen.add(rel_tuple)
                    unique_relationships.append(rel)
            
            return unique_relationships[:10]  # Limit to top 10 relationships
            
        except Exception as e:
            logging.error(f"Error in spaCy relationship extraction: {str(e)}")
            # Fallback to simple extraction
            return self._simple_relationship_extraction(text, entities)