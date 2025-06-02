"""
Comprehensive Benchmarking Suite for Selective Memory Compression System
========================================================================
compression system using real GPT-4o-mini API calls and synthetic data generation.
"""


import os
import sys
import json
import csv
import time
import logging
import random
import statistics
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict, Counter

# Import the memory compression system
sys.path.append('selective_compression_system')
from core.system import SelectiveCompressionSystem
from core.memory import MemorySegment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Data class to store benchmark metrics"""
    # Memory Efficiency
    compression_ratio: float = 0.0
    storage_efficiency: float = 0.0
    hot_memory_usage: float = 0.0
    warm_memory_usage: float = 0.0
    cold_memory_usage: float = 0.0
    
    # Information Preservation
    entity_retention_rate: float = 0.0
    fact_accuracy: float = 0.0
    semantic_similarity: float = 0.0
    bleu_score: float = 0.0
    rouge_score: float = 0.0
    
    # Retrieval Performance
    retrieval_accuracy: float = 0.0
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    context_completeness: float = 0.0
    retrieval_latency: float = 0.0
    
    # Response Quality
    answer_accuracy: float = 0.0
    response_coherence: float = 0.0
    hallucination_rate: float = 0.0
    relevance_score: float = 0.0
    
    # System Performance
    processing_latency: float = 0.0
    api_calls_count: int = 0
    throughput: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Temporal Performance
    memory_decay_rate: float = 0.0
    access_pattern_consistency: float = 0.0
    long_term_retention: float = 0.0


class SyntheticDataGenerator:
    """Generate synthetic conversation data for benchmarking"""
    
    def __init__(self, llm_interface):
        self.llm_interface = llm_interface
        self.conversation_templates = {
            'factual': [
                "Tell me about {topic}",
                "What are the key facts about {topic}?",
                "Can you explain {topic} in detail?",
                "What should I know about {topic}?"
            ],
            'procedural': [
                "How do I {action}?",
                "What are the steps to {action}?",
                "Can you guide me through {action}?",
                "What's the process for {action}?"
            ],
            'personal': [
                "I'm feeling {emotion} about {situation}",
                "I need advice about {situation}",
                "What should I do about {situation}?",
                "I'm struggling with {situation}"
            ],
            'abstract': [
                "What do you think about {concept}?",
                "How would you approach {concept}?",
                "What's your perspective on {concept}?",
                "Can you analyze {concept}?"
            ]
        }
        
        self.topics = {
            'factual': [
                # Science & Technology
                'climate change', 'quantum computing', 'artificial intelligence', 'machine learning',
                'renewable energy', 'space exploration', 'genetics', 'biotechnology', 'nanotechnology',
                'robotics', 'blockchain', 'cybersecurity', 'data science', 'nuclear physics',
                'astronomy', 'neuroscience', 'molecular biology', 'chemistry', 'materials science',
                'bioengineering', 'gene editing', 'stem cells', 'medical research', 'pharmaceuticals',
                'epidemiology', 'public health', 'environmental science', 'oceanography', 'geology',
                'meteorology', 'seismology', 'paleontology', 'archaeology', 'anthropology',
                
                # Social Sciences & Economics
                'economics', 'macroeconomics', 'microeconomics', 'behavioral economics', 
                'international trade', 'monetary policy', 'fiscal policy', 'cryptocurrency',
                'stock market', 'real estate', 'supply chain', 'globalization', 'inflation',
                'psychology', 'cognitive psychology', 'social psychology', 'developmental psychology',
                'sociology', 'political science', 'international relations', 'diplomacy',
                'governance', 'democracy', 'authoritarianism', 'human rights', 'civil liberties',
                'constitutional law', 'criminal justice', 'immigration', 'urban planning',
                
                # History & Culture
                'world history', 'ancient civilizations', 'medieval history', 'renaissance',
                'industrial revolution', 'world wars', 'cold war', 'decolonization',
                'cultural anthropology', 'linguistics', 'literature', 'philosophy',
                'religious studies', 'mythology', 'folklore', 'art history', 'music history',
                'architecture', 'design', 'fashion', 'culinary traditions', 'sports history',
                
                # Geography & Environment
                'physical geography', 'human geography', 'cartography', 'demographics',
                'urbanization', 'rural development', 'natural disasters', 'conservation',
                'biodiversity', 'ecosystems', 'wildlife', 'forestry', 'agriculture',
                'water resources', 'air quality', 'pollution', 'waste management',
                'sustainable development', 'green technology', 'carbon footprint'
            ],
            
            'procedural': [
                # Life Skills
                'cooking pasta', 'baking bread', 'meal planning', 'food preservation',
                'grocery shopping', 'kitchen organization', 'knife skills', 'wine pairing',
                'home brewing', 'gardening', 'composting', 'housekeeping', 'laundry',
                'basic repairs', 'plumbing', 'electrical work', 'painting', 'carpentry',
                'sewing', 'knitting', 'crafting', 'woodworking', 'auto maintenance',
                
                # Learning & Education
                'learning guitar', 'piano lessons', 'singing', 'drawing', 'painting',
                'photography', 'video editing', 'writing', 'speed reading', 'memory techniques',
                'note taking', 'study methods', 'test preparation', 'research skills',
                'critical thinking', 'problem solving', 'language learning', 'public speaking',
                'presentation skills', 'interview preparation', 'networking',
                
                # Technology & Digital
                'writing code', 'web development', 'app development', 'database design',
                'system administration', 'network setup', 'troubleshooting', 'data analysis',
                'digital marketing', 'social media management', 'content creation',
                'video production', 'podcast creation', 'website building', 'SEO',
                'email marketing', 'online selling', 'e-commerce', 'digital security',
                
                # Health & Fitness
                'exercising', 'strength training', 'cardio workouts', 'yoga', 'pilates',
                'stretching', 'running', 'swimming', 'cycling', 'martial arts',
                'meditation', 'mindfulness', 'breathing exercises', 'sleep hygiene',
                'nutrition planning', 'weight management', 'injury prevention',
                'physical therapy', 'mental health care', 'stress management',
                
                # Business & Finance
                'starting a business', 'business planning', 'market research', 'fundraising',
                'accounting', 'bookkeeping', 'tax preparation', 'budgeting', 'investing',
                'retirement planning', 'insurance', 'estate planning', 'credit management',
                'debt reduction', 'saving money', 'negotiation', 'project management',
                'team leadership', 'hiring', 'performance reviews', 'conflict resolution',
                
                # Creative & Artistic
                'creative writing', 'storytelling', 'screenwriting', 'poetry', 'blogging',
                'journaling', 'music composition', 'beat making', 'sound engineering',
                'graphic design', 'logo design', 'illustration', 'animation', '3D modeling',
                'sculpture', 'pottery', 'jewelry making', 'fashion design', 'interior design'
            ],
            
            'personal': [
                # Career & Professional
                'career change', 'job searching', 'resume writing', 'portfolio building',
                'skill development', 'professional networking', 'workplace communication',
                'leadership development', 'work-life balance', 'workplace stress',
                'imposter syndrome', 'career advancement', 'salary negotiation',
                'performance anxiety', 'workplace relationships', 'office politics',
                'remote work', 'freelancing', 'entrepreneurship', 'retirement transition',
                
                # Relationships & Social
                'relationship issues', 'dating', 'marriage', 'parenting', 'family dynamics',
                'friendship', 'social anxiety', 'communication skills', 'conflict resolution',
                'setting boundaries', 'trust issues', 'intimacy', 'breakups', 'divorce',
                'grief', 'loss', 'loneliness', 'social skills', 'networking', 'community building',
                'cultural differences', 'generational gaps', 'peer pressure', 'bullying',
                
                # Mental Health & Wellbeing
                'stress', 'anxiety', 'depression', 'panic attacks', 'trauma', 'PTSD',
                'self-esteem', 'confidence', 'body image', 'perfectionism', 'procrastination',
                'attention deficit', 'memory issues', 'sleep problems', 'addiction',
                'eating disorders', 'mood swings', 'emotional regulation', 'anger management',
                'forgiveness', 'healing', 'therapy', 'counseling', 'self-care', 'mindfulness',
                
                # Life Management
                'time management', 'productivity', 'organization', 'goal setting',
                'habit formation', 'motivation', 'discipline', 'focus', 'decision making',
                'priority setting', 'life planning', 'transitions', 'moving', 'travel planning',
                'financial stress', 'health management', 'chronic illness', 'aging',
                'identity', 'purpose', 'values', 'spirituality', 'personal growth',
                
            ],
            
            'abstract': [
                # Philosophy & Metaphysics
                'consciousness', 'free will', 'determinism', 'mind-body problem', 'personal identity',
                'reality', 'existence', 'being', 'nothingness', 'infinity', 'time',
                'causation', 'possibility', 'necessity', 'universals', 'particulars',
               
                # Ethics & Morality
                'justice', 'fairness', 'equality', 'rights', 'duties', 'virtue', 'vice',
                'good', 'evil', 'moral responsibility', 'punishment', 'forgiveness',
                'compassion', 'empathy', 'altruism', 'selfishness', 'integrity',
               
                # Aesthetics & Beauty
                'beauty', 'ugliness', 'sublime', 'taste', 'aesthetic experience',
                'art', 'creativity', 'imagination', 'inspiration', 'genius',
                'originality', 'authenticity', 'representation', 'expression',
                
                
                # Existential & Spiritual
                'meaning of life', 'purpose', 'suffering', 'death', 'mortality',
                'legacy', 'transcendence', 'spirituality', 'faith', 'doubt',
                'revelation', 'mystery', 'sacred', 'profane', 'divine', 'eternal',
                
                
                # Concepts & Ideas
                'happiness', 'sadness', 'joy', 'melancholy', 'nostalgia', 'longing',
                'desire', 'fulfillment', 'satisfaction', 'contentment', 'peace',
                'chaos', 'order', 'harmony', 'discord', 'balance', 'extremes',
                'moderation', 'excess', 'simplicity', 'complexity', 'clarity',
                
                
            ]
        }   

    def generate_conversation(self, length: int, content_type: str) -> List[Tuple[str, str]]:
        """Generate a synthetic conversation of specified length and type"""
        conversation = []
        
        try:
            # Select topic and templates
            topic_pool = self.topics[content_type]
            template_pool = self.conversation_templates[content_type]
            
            # Generate conversation turns
            context = ""
            for i in range(length):
                # Select topic and template
                topic = random.choice(topic_pool)
                template = random.choice(template_pool)
                
                # Generate user message
                if content_type == 'factual':
                    user_input = template.format(topic=topic)
                elif content_type == 'procedural':
                    user_input = template.format(action=topic)
                elif content_type == 'personal':
                    emotion = random.choice(['anxious', 'excited', 'confused', 'frustrated'])
                    user_input = template.format(emotion=emotion, situation=topic)
                else:  # abstract
                    user_input = template.format(concept=topic)
                
                # Generate assistant response using LLM
                prompt = f"""
                Context: {context}
                
                User: {user_input}
                
                Provide a helpful, detailed response as an AI assistant. Keep the response 
                conversational and informative, around 100-200 words.
                
                Assistant:"""
                
                assistant_response = self.llm_interface.generate(prompt)
                
                # Add to conversation
                conversation.append((user_input, assistant_response))
                
                # Update context for next turn
                context += f"\nUser: {user_input}\nAssistant: {assistant_response}"
                if len(context) > 2000:  # Keep context manageable
                    context = context[-1500:]
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error generating conversation: {str(e)}")
            # Return at least what we have
            if not conversation:
                conversation = [("Hello", "Hi there! How can I help you today?")]
        
        return conversation
    
class MetricsCalculator:
    """Calculate various benchmark metrics"""
    
    def __init__(self, llm_interface):
        self.llm_interface = llm_interface
    
    def calculate_compression_ratio(self, original_segments: List[MemorySegment], 
                                   compressed_segments: List[MemorySegment]) -> float:
        """Calculate compression ratio"""
        if not original_segments or not compressed_segments:
            return 1.0
            
        original_size = sum(len(seg.content) for seg in original_segments)
        compressed_size = sum(len(seg.content) for seg in compressed_segments)
        
        return original_size / max(compressed_size, 1)
    
    def calculate_entity_retention(self, original_content: str, 
                                   compressed_content: str, 
                                   entity_extractor) -> float:
        """Calculate entity retention rate"""
        try:
            original_entities = set(entity_extractor.extract_entities(original_content))
            compressed_entities = set(entity_extractor.extract_entities(compressed_content))
            
            if not original_entities:
                return 1.0
                
            retained = original_entities.intersection(compressed_entities)
            return len(retained) / len(original_entities)
        except:
            return 0.0
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        try:
            embedding1 = self.llm_interface.embed_text(text1)
            embedding2 = self.llm_interface.embed_text(text2)
            
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            
            return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0
        except:
            return 0.0
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score (simplified version)"""
        try:
            ref_words = reference.lower().split()
            cand_words = candidate.lower().split()
            
            if not ref_words or not cand_words:
                return 0.0
            
            # 1-gram precision
            ref_counts = Counter(ref_words)
            cand_counts = Counter(cand_words)
            
            overlap = sum(min(ref_counts[word], cand_counts[word]) 
                         for word in cand_counts)
            
            precision = overlap / len(cand_words)
            
            # Brevity penalty
            bp = min(1.0, len(cand_words) / len(ref_words))
            
            return bp * precision
        except:
            return 0.0
    
    def calculate_rouge_score(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L score (simplified version)"""
        try:
            ref_words = reference.lower().split()
            cand_words = candidate.lower().split()
            
            if not ref_words or not cand_words:
                return 0.0
            
            # Find LCS
            lcs_length = self._lcs_length(ref_words, cand_words)
            
            precision = lcs_length / len(cand_words)
            recall = lcs_length / len(ref_words)
            
            if precision + recall == 0:
                return 0.0
                
            f1 = 2 * precision * recall / (precision + recall)
            return f1
        except:
            return 0.0
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def evaluate_response_quality(self, question: str, answer: str, 
                                  reference_answer: str = None) -> Dict[str, float]:
        """Evaluate response quality using LLM"""
        try:
            # Generate evaluation prompt
            prompt = f"""
            Evaluate the following AI assistant response on a scale of 0-10:
            
            Question: {question}
            Response: {answer}
            
            Rate the response on:
            1. Accuracy (0-10): How factually correct is the response?
            2. Coherence (0-10): How well-structured and logical is the response?
            3. Relevance (0-10): How well does it address the question?
            4. Completeness (0-10): How comprehensive is the response?
            
            Provide your ratings in this exact format:
            Accuracy: X
            Coherence: X
            Relevance: X
            Completeness: X
            """
            
            evaluation = self.llm_interface.generate(prompt)
            
            # Parse ratings
            ratings = {}
            for line in evaluation.split('\n'):
                line = line.strip()
                if ':' in line:
                    metric, score = line.split(':', 1)
                    try:
                        ratings[metric.strip().lower()] = float(score.strip()) / 10.0
                    except:
                        continue
            
            return {
                'accuracy': ratings.get('accuracy', 0.5),
                'coherence': ratings.get('coherence', 0.5),
                'relevance': ratings.get('relevance', 0.5),
                'completeness': ratings.get('completeness', 0.5)
            }
        except:
            return {'accuracy': 0.5, 'coherence': 0.5, 'relevance': 0.5, 'completeness': 0.5}
        

class MemoryCompressionBenchmark:
    """Main benchmarking class"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
            
        # Initialize system components
        self.system = SelectiveCompressionSystem(
            model_name="gpt-4o-mini",
            hot_size=4000,
            warm_size=16000,
            cold_size=64000,
            api_key=self.api_key
        )
        
        self.data_generator = SyntheticDataGenerator(self.system.llm_interface)
        self.metrics_calculator = MetricsCalculator(self.system.llm_interface)
        
        # Benchmark results storage
        self.results = []
        self.detailed_results = defaultdict(list)
        
        logger.info("Benchmark system initialized")
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite"""
        logger.info("Starting comprehensive benchmark...")
        
        start_time = time.time()
        
        # Test scenarios
        test_scenarios = [
            # (length, content_type, name)
            (5, 'factual', 'short_factual'),
            (10, 'factual', 'medium_factual'),
            (15, 'factual', 'long_factual'),
            (5, 'procedural', 'short_procedural'),
            (20, 'procedural', 'medium_procedural'),
            (5, 'personal', 'short_personal'),
            (15, 'personal', 'medium_personal'),
            (5, 'abstract', 'short_abstract'),
            (10, 'abstract', 'medium_abstract'),
        ]
        
        # Run each test scenario
        for length, content_type, scenario_name in test_scenarios:
            logger.info(f"Running scenario: {scenario_name}")
            try:
                scenario_results = self._run_scenario(length, content_type, scenario_name)
                self.results.append(scenario_results)
                self.detailed_results[scenario_name] = scenario_results
            except Exception as e:
                logger.error(f"Error in scenario {scenario_name}: {str(e)}")
        
        # Run stress tests
        logger.info("Running stress tests...")
        try:
            stress_results = self._run_stress_tests()
            self.results.extend(stress_results)
        except Exception as e:
            logger.error(f"Error in stress tests: {str(e)}")
        
        # Compile final results
        total_time = time.time() - start_time
        final_results = self._compile_results(total_time)
        
        logger.info(f"Benchmark completed in {total_time:.2f} seconds")
        return final_results
    
    def _run_scenario(self, length: int, content_type: str, scenario_name: str) -> Dict[str, Any]:
        """Run a single test scenario"""
        logger.info(f"Generating {length}-turn {content_type} conversation...")
        
        # Reset system state
        self.system.reset_memory()
        
        # Generate conversation
        conversation = self.data_generator.generate_conversation(length, content_type)
        
        # Track metrics during conversation
        metrics = BenchmarkMetrics()
        conversation_start = time.time()
        api_calls_start = self._count_api_calls()
        
        original_segments = []
        responses = []
        
        # Process conversation
        for i, (user_input, expected_response) in enumerate(conversation):
            turn_start = time.time()
            
            # Store original content before compression
            if i > 0:  # Skip first turn
                hot_segments = self.system.memory_manager.hot_memory.get_all_segments()
                original_segments.extend([seg for seg in hot_segments])
            
            # Process input
            actual_response = self.system.process_input(user_input)
            responses.append((user_input, actual_response, expected_response))
            
            # Update metrics
            turn_latency = time.time() - turn_start
            metrics.processing_latency += turn_latency
            
            # Force memory management periodically
            if i % 3 == 0:
                self.system.force_memory_management()
        
        # Calculate final metrics
        conversation_time = time.time() - conversation_start
        api_calls_end = self._count_api_calls()
        
        # Memory efficiency metrics
        memory_stats = self.system.get_memory_stats()
        # Convert utilization to decimal (0-1 range) for consistency
        metrics.hot_memory_usage = memory_stats['hot_memory']['utilization'] / 100.0
        metrics.warm_memory_usage = memory_stats['warm_memory']['utilization'] / 100.0
        metrics.cold_memory_usage = memory_stats['cold_memory']['utilization'] / 100.0
                
        # Calculate compression ratios
        compressed_segments = []
        compressed_segments.extend(self.system.memory_manager.warm_memory.get_all_segments())
        compressed_segments.extend(self.system.memory_manager.cold_memory.get_all_segments())
        
        if original_segments and compressed_segments:
            metrics.compression_ratio = self.metrics_calculator.calculate_compression_ratio(
                original_segments, compressed_segments
            )
        
        # Information preservation metrics
        self._calculate_preservation_metrics(metrics, original_segments, compressed_segments)
        
        # Response quality metrics
        self._calculate_response_quality_metrics(metrics, responses)
        
        # System performance metrics
        metrics.api_calls_count = api_calls_end - api_calls_start
        metrics.throughput = len(conversation) / conversation_time
        metrics.processing_latency /= len(conversation)  # Average latency
        
        # Retrieval performance metrics
        self._calculate_retrieval_metrics(metrics, conversation)
        
        return {
            'scenario_name': scenario_name,
            'metrics': asdict(metrics),
            'conversation_length': length,
            'content_type': content_type,
            'total_time': conversation_time,
            'memory_stats': memory_stats,
            'responses': responses[:3]  # Sample responses for analysis
        }
    
    def _calculate_preservation_metrics(self, metrics: BenchmarkMetrics, 
                                        original_segments: List[MemorySegment],
                                        compressed_segments: List[MemorySegment]):
        """Calculate information preservation metrics"""
        if not original_segments or not compressed_segments:
            return
            
        try:
            # Calculate entity retention
            original_content = " ".join([seg.content for seg in original_segments])
            compressed_content = " ".join([seg.content for seg in compressed_segments])
            
            metrics.entity_retention_rate = self.metrics_calculator.calculate_entity_retention(
                original_content, compressed_content, self.system.entity_extractor
            )
            
            # Calculate semantic similarity
            metrics.semantic_similarity = self.metrics_calculator.calculate_semantic_similarity(
                original_content[:1000], compressed_content[:1000]  # Limit for API efficiency
            )
            
            # Calculate BLEU and ROUGE scores
            metrics.bleu_score = self.metrics_calculator.calculate_bleu_score(
                original_content, compressed_content
            )
            metrics.rouge_score = self.metrics_calculator.calculate_rouge_score(
                original_content, compressed_content
            )
            
        except Exception as e:
            logger.error(f"Error calculating preservation metrics: {str(e)}")
    
    def _calculate_response_quality_metrics(self, metrics: BenchmarkMetrics, 
                                            responses: List[Tuple[str, str, str]]):
        """Calculate response quality metrics"""
        if not responses:
            return
            
        quality_scores = []
        
        try:
            # Sample a few responses for evaluation
            sample_responses = responses[:min(3, len(responses))]
            
            for user_input, actual_response, expected_response in sample_responses:
                quality = self.metrics_calculator.evaluate_response_quality(
                    user_input, actual_response, expected_response
                )
                quality_scores.append(quality)
            
            # Average the scores
            if quality_scores:
                metrics.answer_accuracy = statistics.mean([q['accuracy'] for q in quality_scores])
                metrics.response_coherence = statistics.mean([q['coherence'] for q in quality_scores])
                metrics.relevance_score = statistics.mean([q['relevance'] for q in quality_scores])
                
                # Estimate hallucination rate (inverse of accuracy)
                metrics.hallucination_rate = 1.0 - metrics.answer_accuracy
                
        except Exception as e:
            logger.error(f"Error calculating response quality metrics: {str(e)}")
    
    def _calculate_retrieval_metrics(self, metrics: BenchmarkMetrics, 
                                     conversation: List[Tuple[str, str]]):
        """Calculate retrieval performance metrics"""
        if len(conversation) < 3:
            return
            
        try:
            # Test retrieval with a sample query
            sample_query = conversation[-1][0]  # Last user input
            
            retrieval_start = time.time()
            
            # Get current context
            current_context = self.system.integration_layer._get_current_context()
            
            # Identify relevant segments
            relevant_segments = self.system.retrieval_engine.identify_relevant_segments(
                current_context, sample_query
            )
            
            retrieval_time = time.time() - retrieval_start
            metrics.retrieval_latency = retrieval_time
            
            # Calculate retrieval accuracy (simplified)
            total_segments = (len(self.system.memory_manager.warm_memory.segments) + 
                             len(self.system.memory_manager.cold_memory.segments))
            
            if total_segments > 0:
                metrics.retrieval_precision = len(relevant_segments) / max(total_segments, 1)
                metrics.retrieval_recall = min(1.0, len(relevant_segments) / max(3, 1))  # Assume 3 relevant
                
                if metrics.retrieval_precision + metrics.retrieval_recall > 0:
                    metrics.retrieval_accuracy = (2 * metrics.retrieval_precision * metrics.retrieval_recall) / \
                                                (metrics.retrieval_precision + metrics.retrieval_recall)
            
        except Exception as e:
            logger.error(f"Error calculating retrieval metrics: {str(e)}")
    
    def _run_stress_tests(self) -> List[Dict[str, Any]]:
        """Run stress tests at capacity limits"""
        stress_results = []
        
        # Test 1: Memory capacity stress test
        logger.info("Running memory capacity stress test...")
        try:
            self.system.reset_memory()
            
            # Generate a very long conversation to fill memory
            long_conversation = self.data_generator.generate_conversation(50, 'factual')
            
            start_time = time.time()
            for user_input, _ in long_conversation:
                self.system.process_input(user_input)
            
            stress_time = time.time() - start_time
            final_stats = self.system.get_memory_stats()
            
            stress_results.append({
                'test_name': 'memory_capacity_stress',
                'conversation_length': 50,
                'total_time': stress_time,
                'final_memory_stats': final_stats,
                'throughput': 50 / stress_time
            })
            
        except Exception as e:
            logger.error(f"Memory capacity stress test failed: {str(e)}")
        
        # Test 2: Rapid access pattern test
        logger.info("Running rapid access pattern test...")
        try:
            self.system.reset_memory()
            
            # Create initial conversation
            base_conversation = self.data_generator.generate_conversation(20, 'mixed')
            for user_input, _ in base_conversation:
                self.system.process_input(user_input)
            
            # Rapid fire queries
            rapid_queries = [
                "What did we talk about earlier?",
                "Can you remind me about the main points?",
                "What was mentioned about the first topic?",
                "Summarize our conversation so far"
            ] * 5
            
            start_time = time.time()
            for query in rapid_queries:
                self.system.process_input(query)
            
            rapid_time = time.time() - start_time
            
            stress_results.append({
                'test_name': 'rapid_access_pattern',
                'query_count': len(rapid_queries),
                'total_time': rapid_time,
                'avg_response_time': rapid_time / len(rapid_queries),
                'throughput': len(rapid_queries) / rapid_time
            })
            
        except Exception as e:
            logger.error(f"Rapid access pattern test failed: {str(e)}")
        
        return stress_results
    
    def _count_api_calls(self) -> int:
        """Rough estimate of API calls made (simplified tracking)"""
        # This is a simplified implementation
        # In practice, you'd want to instrument the LLM interface
        return 0
    
    def _compile_results(self, total_time: float) -> Dict[str, Any]:
        """Compile and analyze all benchmark results"""
        if not self.results:
            return {'error': 'No results to compile'}
        
        # Extract metrics from all scenarios
        all_metrics = []
        for result in self.results:
            if 'metrics' in result:
                cleaned_metrics = {}
                for key, value in result['metrics'].items():
                    try:
                        # Ensure all metric values are numeric
                        if isinstance(value, str):
                            cleaned_metrics[key] = float(value) if value.replace('.', '').replace('-', '').isdigit() else 0.0
                        elif value is None:
                            cleaned_metrics[key] = 0.0
                        else:
                            cleaned_metrics[key] = float(value)
                    except (ValueError, TypeError):
                        cleaned_metrics[key] = 0.0
                all_metrics.append(cleaned_metrics)
        
        if not all_metrics:
            return {'error': 'No metrics to analyze'}
        
        # Calculate aggregate statistics
        aggregate_stats = {}
        metric_names = list(all_metrics[0].keys())
        
        for metric_name in metric_names:
            # Filter out None values AND ensure all values are numeric
            values = []
            for m in all_metrics:
                val = m.get(metric_name)
                if val is not None:
                    try:
                        # Convert to float to ensure it's numeric
                        numeric_val = float(val)
                        # Skip NaN and infinite values
                        if not (np.isnan(numeric_val) or np.isinf(numeric_val)):
                            values.append(numeric_val)
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        continue
            
            if values:
                try:
                    aggregate_stats[metric_name] = {
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
                except Exception as e:
                    logger.warning(f"Error calculating stats for {metric_name}: {str(e)}")
                    # Provide default values
                    aggregate_stats[metric_name] = {
                        'mean': 0.0,
                        'median': 0.0,
                        'std_dev': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'count': 0
                    }
        
        # Performance summary
        performance_summary = {
            'total_scenarios': len(self.results),
            'total_time': total_time,
            'avg_scenario_time': total_time / len(self.results),
            'overall_throughput': sum(r.get('throughput', 0) for r in self.results if 'throughput' in r) / len(self.results)
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_summary': performance_summary,
            'aggregate_metrics': aggregate_stats,
            'detailed_results': self.detailed_results,
            'raw_results': self.results
        }
    
    def export_results(self, results: Dict[str, Any], output_dir: str = "benchmark_results"):
        """Export results to CSV and JSON formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export to JSON
        json_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Export aggregate metrics to CSV
        csv_path = os.path.join(output_dir, f"aggregate_metrics_{timestamp}.csv")
        with open(csv_path, 'w', newline='') as f:
            if 'aggregate_metrics' in results:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Mean', 'Median', 'Std_Dev', 'Min', 'Max', 'Count'])
                
                for metric_name, stats in results['aggregate_metrics'].items():
                    writer.writerow([
                        metric_name,
                        stats['mean'],
                        stats['median'],
                        stats['std_dev'],
                        stats['min'],
                        stats['max'],
                        stats['count']
                    ])
        
        # Export detailed scenario results to CSV
        scenarios_csv_path = os.path.join(output_dir, f"scenario_results_{timestamp}.csv")
        with open(scenarios_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['Scenario', 'Length', 'Type', 'Total_Time']
            if self.results and 'metrics' in self.results[0]:
                header.extend(self.results[0]['metrics'].keys())
            writer.writerow(header)
            
            # Write data
            for result in self.results:
                if 'metrics' in result:
                    row = [
                        result.get('scenario_name', 'unknown'),
                        result.get('conversation_length', 0),
                        result.get('content_type', 'unknown'),
                        result.get('total_time', 0)
                    ]
                    row.extend(result['metrics'].values())
                    writer.writerow(row)
        
        logger.info(f"Results exported to {output_dir}")
        return {
            'json_path': json_path,
            'csv_path': csv_path,
            'scenarios_csv_path': scenarios_csv_path
        }
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive benchmark report"""
        report = []
        report.append("=" * 80)
        report.append("SELECTIVE MEMORY COMPRESSION SYSTEM - BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {results.get('timestamp', 'Unknown')}")
        report.append("")
        
        # Performance Summary
        if 'performance_summary' in results:
            perf = results['performance_summary']
            report.append("PERFORMANCE SUMMARY")
            report.append("-" * 20)
            report.append(f"Total Scenarios Tested: {perf.get('total_scenarios', 0)}")
            report.append(f"Total Execution Time: {perf.get('total_time', 0):.2f} seconds")
            report.append(f"Average Time per Scenario: {perf.get('avg_scenario_time', 0):.2f} seconds")
            report.append(f"Overall Throughput: {perf.get('overall_throughput', 0):.2f} operations/second")
            report.append("")
        
        # Aggregate Metrics Analysis
        if 'aggregate_metrics' in results:
            metrics = results['aggregate_metrics']
            report.append("AGGREGATE METRICS ANALYSIS")
            report.append("-" * 30)
            
            # Memory Efficiency Section
            report.append("\n📊 MEMORY EFFICIENCY")
            memory_metrics = ['compression_ratio', 'storage_efficiency', 'hot_memory_usage', 
                            'warm_memory_usage', 'cold_memory_usage']
            for metric in memory_metrics:
                if metric in metrics:
                    stats = metrics[metric]
                    report.append(f"  {metric.replace('_', ' ').title()}: "
                                f"{stats['mean']:.3f} ± {stats['std_dev']:.3f} "
                                f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
            
            # Information Preservation Section
            report.append("\n🧠 INFORMATION PRESERVATION")
            preservation_metrics = ['entity_retention_rate', 'fact_accuracy', 'semantic_similarity', 
                                'bleu_score', 'rouge_score']
            for metric in preservation_metrics:
                if metric in metrics:
                    stats = metrics[metric]
                    report.append(f"  {metric.replace('_', ' ').title()}: "
                                f"{stats['mean']:.3f} ± {stats['std_dev']:.3f} "
                                f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
            
            # Retrieval Performance Section
            report.append("\n🔍 RETRIEVAL PERFORMANCE")
            retrieval_metrics = ['retrieval_accuracy', 'retrieval_precision', 'retrieval_recall', 
                                'context_completeness', 'retrieval_latency']
            for metric in retrieval_metrics:
                if metric in metrics:
                    stats = metrics[metric]
                    unit = " ms" if "latency" in metric else ""
                    report.append(f"  {metric.replace('_', ' ').title()}: "
                                f"{stats['mean']:.3f}{unit} ± {stats['std_dev']:.3f} "
                                f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
            
            # Response Quality Section
            report.append("\n✨ RESPONSE QUALITY")
            quality_metrics = ['answer_accuracy', 'response_coherence', 'hallucination_rate', 
                            'relevance_score']
            for metric in quality_metrics:
                if metric in metrics:
                    stats = metrics[metric]
                    report.append(f"  {metric.replace('_', ' ').title()}: "
                                f"{stats['mean']:.3f} ± {stats['std_dev']:.3f} "
                                f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
            
            # System Performance Section
            report.append("\n⚡ SYSTEM PERFORMANCE")
            system_metrics = ['processing_latency', 'api_calls_count', 'throughput', 'memory_usage_mb']
            for metric in system_metrics:
                if metric in metrics:
                    stats = metrics[metric]
                    if "latency" in metric:
                        unit = " ms"
                    elif "memory" in metric:
                        unit = " MB"
                    elif "throughput" in metric:
                        unit = " ops/sec"
                    elif "calls" in metric:
                        unit = " calls"
                    else:
                        unit = ""
                    
                    report.append(f"  {metric.replace('_', ' ').title()}: "
                                f"{stats['mean']:.3f}{unit} ± {stats['std_dev']:.3f} "
                                f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
        
        # Scenario Breakdown
        if 'detailed_results' in results:
            report.append("\n" + "=" * 40)
            report.append("DETAILED SCENARIO RESULTS")
            report.append("=" * 40)
            
            for scenario_name, scenario_data in results['detailed_results'].items():
                if isinstance(scenario_data, dict):
                    report.append(f"\n📋 {scenario_name.upper().replace('_', ' ')}")
                    report.append("-" * (len(scenario_name) + 5))
                    report.append(f"  Conversation Length: {scenario_data.get('conversation_length', 'N/A')}")
                    report.append(f"  Content Type: {scenario_data.get('content_type', 'N/A')}")
                    report.append(f"  Execution Time: {scenario_data.get('total_time', 0):.2f}s")
                    
                    if 'metrics' in scenario_data:
                        metrics = scenario_data['metrics']
                        # Highlight key metrics for this scenario
                        key_metrics = ['compression_ratio', 'semantic_similarity', 'answer_accuracy', 
                                    'retrieval_accuracy', 'processing_latency']
                        
                        for metric in key_metrics:
                            if metric in metrics and metrics[metric] is not None:
                                value = metrics[metric]
                                unit = " ms" if "latency" in metric else ""
                                report.append(f"    {metric.replace('_', ' ').title()}: {value:.3f}{unit}")
                    
                    # Memory utilization for this scenario
                    if 'memory_stats' in scenario_data:
                        mem_stats = scenario_data['memory_stats']
                        report.append("  Memory Utilization:")
                        for tier in ['hot_memory', 'warm_memory', 'cold_memory']:
                            if tier in mem_stats:
                                util = mem_stats[tier].get('utilization', 0)
                                # Handle both numeric and string percentage formats
                                if isinstance(util, str):
                                    util_clean = float(util.rstrip('%')) / 100.0
                                else:
                                    util_clean = util / 100.0 if util > 1 else util
                                report.append(f"    {tier.replace('_', ' ').title()}: {util_clean:.1%}")
        
        # Recommendations Section
        report.append("\n" + "=" * 40)
        report.append("RECOMMENDATIONS & INSIGHTS")
        report.append("=" * 40)
        
        recommendations = []
        
        # Analyze compression efficiency
        if 'aggregate_metrics' in results and 'compression_ratio' in results['aggregate_metrics']:
            comp_ratio = results['aggregate_metrics']['compression_ratio']['mean']
            if comp_ratio < 2.0:
                recommendations.append("🔧 Low compression ratio detected. Consider adjusting compression thresholds or algorithms.")
            elif comp_ratio > 5.0:
                recommendations.append("✅ Excellent compression ratio achieved. System is efficiently reducing memory usage.")
            else:
                recommendations.append("👍 Good compression ratio. System is balancing efficiency and information retention well.")
        
        # Analyze information preservation
        if 'aggregate_metrics' in results and 'semantic_similarity' in results['aggregate_metrics']:
            sem_sim = results['aggregate_metrics']['semantic_similarity']['mean']
            if sem_sim < 0.7:
                recommendations.append("⚠️ Low semantic similarity after compression. Information loss may be too high.")
            elif sem_sim > 0.9:
                recommendations.append("✅ Excellent semantic preservation. Compression maintains meaning well.")
            else:
                recommendations.append("👍 Good semantic preservation. Acceptable information retention.")
        
        # Analyze response quality
        if 'aggregate_metrics' in results and 'answer_accuracy' in results['aggregate_metrics']:
            accuracy = results['aggregate_metrics']['answer_accuracy']['mean']
            if accuracy < 0.7:
                recommendations.append("⚠️ Low response accuracy. Memory compression may be affecting answer quality.")
            elif accuracy > 0.9:
                recommendations.append("✅ High response accuracy maintained despite compression.")
            else:
                recommendations.append("👍 Acceptable response accuracy. Minor impact from compression.")
        
        # Analyze system performance
        if 'aggregate_metrics' in results and 'processing_latency' in results['aggregate_metrics']:
            latency = results['aggregate_metrics']['processing_latency']['mean']
            if latency > 2.0:
                recommendations.append("⚡ High processing latency detected. Consider optimizing retrieval algorithms.")
            elif latency < 0.5:
                recommendations.append("✅ Excellent response times. System is highly optimized.")
            else:
                recommendations.append("👍 Acceptable processing latency for the complexity of operations.")
        
        # Add recommendations to report
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        
        if not recommendations:
            report.append("📊 Analysis complete. All metrics within expected ranges.")
        
        # System Configuration Summary
        report.append("\n" + "=" * 40)
        report.append("SYSTEM CONFIGURATION")
        report.append("=" * 40)
        report.append("🔧 Memory Tiers:")
        report.append("   • Hot Memory: 4,000 tokens (immediate access)")
        report.append("   • Warm Memory: 16,000 tokens (compressed, fast retrieval)")
        report.append("   • Cold Memory: 64,000 tokens (highly compressed, slower retrieval)")
        report.append("🤖 Model: GPT-4o-mini")
        report.append("📊 Compression: Selective semantic compression with entity preservation")
        
        # Footer
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main function to run the benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Memory Compression Benchmark')
    parser.add_argument('--api-key', help='OpenAI API key (or use OPENAI_API_KEY env var)')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory for results')
    parser.add_argument('--export-only', action='store_true', help='Only export existing results without running benchmark')
    
    args = parser.parse_args()
    
    try:
        # Initialize benchmark
        benchmark = MemoryCompressionBenchmark(api_key=args.api_key)
        
        if not args.export_only:
            # Run comprehensive benchmark
            print("🚀 Starting comprehensive benchmark suite...")
            print("This may take several minutes depending on API response times.")
            print("-" * 60)
            
            results = benchmark.run_comprehensive_benchmark()
            
            # Export results
            export_paths = benchmark.export_results(results, args.output_dir)
            
            # Generate and save report
            report = benchmark.generate_report(results)
            report_path = os.path.join(args.output_dir, f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            print("\n" + "=" * 60)
            print("✅ BENCHMARK COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"📊 Results exported to: {args.output_dir}")
            print(f"📄 Detailed report: {report_path}")
            print(f"📈 CSV data: {export_paths['csv_path']}")
            print(f"🗂️ JSON data: {export_paths['json_path']}")
            print("\nReport Preview:")
            print("-" * 30)
            print(report[:1000] + "..." if len(report) > 1000 else report)
            
        else:
            print("Export-only mode: Please ensure benchmark results exist in the specified directory.")
    
    except KeyboardInterrupt:
        print("\n⏹️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        print(f"❌ Benchmark failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()