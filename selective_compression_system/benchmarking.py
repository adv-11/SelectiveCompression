"""
Selective Compression System Research Benchmarking

This script benchmarks the Selective Compression Memory System, testing its performance
under various scenarios and collecting metrics related to compression ratios,
memory utilization, response quality, and system behavior.
"""

import os
import time
import json
import random
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

# Import system components
from core.system import SelectiveCompressionSystem
from core.memory import MemorySegment
from modules.entity_extractor import EntityExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResearchBenchmark:
    """Benchmark class for testing Selective Compression System."""
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini", 
                 hot_size: int = 2000, 
                 warm_size: int = 8000, 
                 cold_size: int = 32000,
                 api_key: Optional[str] = None):
        """Initialize benchmarking environment.
        
        Args:
            model_name: LLM model to use
            hot_size: Hot memory size in tokens
            warm_size: Warm memory size in tokens
            cold_size: Cold memory size in tokens
            api_key: API key for the LLM service
        """
        self.model_name = model_name
        self.hot_size = hot_size
        self.warm_size = warm_size
        self.cold_size = cold_size
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("No API key provided. Set OPENAI_API_KEY env variable or pass it as argument.")
        
        # Initialize system
        self.system = SelectiveCompressionSystem(
            model_name=model_name,
            hot_size=hot_size,
            warm_size=warm_size,
            cold_size=cold_size,
            api_key=self.api_key
        )
        
        # Storage for benchmark results
        self.results = {
            "compression_ratios": [],
            "memory_utilization": [],
            "response_times": [],
            "retrieval_quality": [],
            "system_behavior": [],
            "entity_preservation": []
        }
        
        # Setup output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"benchmark_results_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized benchmark with model {model_name}")
        logger.info(f"Memory config: hot={hot_size}, warm={warm_size}, cold={cold_size}")
        
    def load_test_data(self, filepath: str) -> List[Dict[str, str]]:
        """Load test conversation data from file.
        
        Args:
            filepath: Path to test data file (JSON format)
            
        Returns:
            List of conversation turns
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} conversation turns from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            # Generate synthetic data as fallback
            logger.info("Generating synthetic test data as fallback")
            return self.generate_synthetic_data(100)
    
    def generate_synthetic_data(self, num_turns: int) -> List[Dict[str, str]]:
        """Generate synthetic conversation data for testing.
        
        Args:
            num_turns: Number of conversation turns to generate
            
        Returns:
            List of conversation turns
        """
        topics = ["machine learning", "climate change", "space exploration", 
                 "renewable energy", "quantum computing", "artificial intelligence",
                 "virtual reality", "blockchain technology", "genetic engineering",
                 "robotics", "cybersecurity", "nanotechnology"]
        
        questions = [
            "What are the latest developments in {topic}?",
            "How does {topic} impact daily life?",
            "Can you explain the basic principles of {topic}?",
            "What are the ethical concerns related to {topic}?",
            "How has {topic} evolved over the past decade?",
            "What's the relationship between {topic} and sustainability?",
            "Who are the leading experts in {topic}?",
            "Can you compare {topic} with similar technologies?",
            "What are the limitations of current {topic} approaches?",
            "How might {topic} change in the next 5 years?"
        ]
        
        follow_ups = [
            "Tell me more about that.",
            "Why is that significant?",
            "How does that compare to previous approaches?",
            "What are the practical applications?",
            "Are there any risks associated with that?",
            "Who is working on addressing those challenges?",
            "Can you elaborate on the technical details?",
            "How would that affect the average person?",
            "What's your assessment of this development?",
            "Are there alternative perspectives on this issue?"
        ]
        
        data = []
        current_topic = random.choice(topics)
        topic_turns = 0
        max_topic_turns = random.randint(3, 8)
        
        for i in range(num_turns):
            # Occasionally change topic
            if topic_turns >= max_topic_turns:
                current_topic = random.choice(topics)
                topic_turns = 0
                max_topic_turns = random.randint(3, 8)
            
            if topic_turns == 0:
                # Start new topic with a question
                user_input = random.choice(questions).format(topic=current_topic)
            else:
                # Follow up on existing topic
                user_input = random.choice(follow_ups)
            
            data.append({"role": "user", "content": user_input})
            topic_turns += 1
        
        return data
    
    def run_memory_test(self, test_data: List[Dict[str, str]]) -> None:
        """Run memory test with conversation data.
        
        Args:
            test_data: List of conversation turns
        """
        logger.info("Starting memory test...")
        
        memory_stats = []
        compression_events = []
        retrieval_events = []
        response_times = []
        
        # Process each conversation turn
        for i, turn in enumerate(tqdm(test_data, desc="Processing turns")):
            if turn["role"] != "user":
                continue
                
            # Process user input and measure response time
            start_time = time.time()
            response = self.system.process_input(turn["content"])
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Collect memory stats after each turn
            stats = self.system.get_memory_stats()
            memory_stats.append({
                "turn": i,
                "hot_size": stats["hot_memory"]["size"],
                "warm_size": stats["warm_memory"]["size"],
                "cold_size": stats["cold_memory"]["size"],
                "hot_segments": stats["hot_memory"]["segment_count"],
                "warm_segments": stats["warm_memory"]["segment_count"],
                "cold_segments": stats["cold_memory"]["segment_count"],
                "response_time": response_time
            })
            
            # Force memory management every 10 turns to simulate pressure
            if i > 0 and i % 10 == 0:
                logger.info(f"Forcing memory management at turn {i}")
                management_results = self.system.force_memory_management()
                
                # Track compression events
                if "before" in management_results and "after" in management_results:
                    before = management_results["before"]
                    after = management_results["after"]
                    
                    # Calculate changes
                    hot_change = before["hot_memory"]["segment_count"] - after["hot_memory"]["segment_count"]
                    warm_change = after["warm_memory"]["segment_count"] - before["warm_memory"]["segment_count"]
                    cold_change = after["cold_memory"]["segment_count"] - before["cold_memory"]["segment_count"]
                    
                    if hot_change > 0:  # Compression happened
                        compression_events.append({
                            "turn": i,
                            "hot_to_warm": hot_change,
                            "warm_to_cold": warm_change,
                            "evicted": max(0, warm_change - cold_change)
                        })
        
        # Save results
        self.results["memory_utilization"] = memory_stats
        self.results["compression_events"] = compression_events
        self.results["response_times"] = response_times
        
        # Save raw data
        pd.DataFrame(memory_stats).to_csv(
            os.path.join(self.output_dir, "memory_stats.csv"), 
            index=False
        )
        
        logger.info("Memory test completed.")
    
    def run_compression_ratio_test(self) -> None:
        """Test compression ratios across different content types."""
        logger.info("Starting compression ratio test...")
        
        # Define test content of different types
        test_contents = {
            "factual": [
                "The speed of light in a vacuum is 299,792,458 meters per second. This constant, denoted by 'c', is a fundamental physical constant. It plays a crucial role in many areas of physics, including Einstein's theory of special relativity, which states that the speed of light is the same for all observers, regardless of their relative motion or the motion of the light source.",
                "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China–Nepal border runs across its summit point. Its elevation of 8,848.86 m was most recently established in 2020 by the Chinese and Nepali authorities."
            ],
            "conversational": [
                "User: Can you help me understand how neural networks work?\nAssistant: Of course! Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of artificial neurons organized in layers that can learn patterns from data. The basic structure includes an input layer, one or more hidden layers, and an output layer. Each connection between neurons has a weight that adjusts during training to minimize error.",
                "User: What's the difference between machine learning and deep learning?\nAssistant: Great question! Machine learning is a broader field focusing on algorithms that improve through experience. Deep learning is actually a subset of machine learning that specifically uses neural networks with multiple layers (hence 'deep'). While traditional machine learning often requires feature engineering, deep learning automatically discovers relevant features in the data through its layered structure."
            ],
            "narrative": [
                "The old lighthouse stood lonely against the darkening sky, its beam cutting through the approaching storm. Captain Morris had seen many such nights during his fifty years at sea, but something about this one felt different. The waves crashed against the rocky shore with unusual force, as if driven by some ancient anger.",
                "Sarah gazed at the constellation Orion from her balcony, remembering how her grandfather had taught her to find it when she was just seven years old. 'The stars will always guide you home,' he'd said. Now, twenty years later and a thousand miles from where she grew up, those same stars brought a comforting familiarity to an otherwise foreign sky."
            ]
        }
        
        results = []
        
        # Test each content type
        for content_type, contents in test_contents.items():
            for content in contents:
                # Create a test segment
                segment = MemorySegment(
                    content=content,
                    id=f"test_{content_type}_{len(content)%100}",
                    creation_time=time.time(),
                    importance=0.5
                )
                
                # Get original size
                original_size = segment.get_size()
                
                # Compress at level 1
                compressed_l1 = self.system.compressor.compress(segment, level=1)
                l1_size = compressed_l1.get_size()
                l1_ratio = original_size / l1_size if l1_size > 0 else float('inf')
                
                # Compress at level 2
                compressed_l2 = self.system.compressor.compress(segment, level=2)
                l2_size = compressed_l2.get_size()
                l2_ratio = original_size / l2_size if l2_size > 0 else float('inf')
                
                # Entity preservation test
                original_entities = self.system.entity_extractor.extract_entities(content)
                l1_entities = self.system.entity_extractor.extract_entities(compressed_l1.content)
                l2_entities = self.system.entity_extractor.extract_entities(compressed_l2.content)
                
                # Calculate entity preservation ratios
                l1_entity_preservation = len(set(l1_entities).intersection(set(original_entities))) / len(original_entities) if original_entities else 1.0
                l2_entity_preservation = len(set(l2_entities).intersection(set(original_entities))) / len(original_entities) if original_entities else 1.0
                
                # Record results
                results.append({
                    "content_type": content_type,
                    "original_size": original_size,
                    "l1_size": l1_size,
                    "l2_size": l2_size,
                    "l1_ratio": l1_ratio,
                    "l2_ratio": l2_ratio,
                    "original_entities": len(original_entities),
                    "l1_entities": len(l1_entities),
                    "l2_entities": len(l2_entities),
                    "l1_entity_preservation": l1_entity_preservation,
                    "l2_entity_preservation": l2_entity_preservation
                })
        
        # Save results
        self.results["compression_ratios"] = results
        pd.DataFrame(results).to_csv(
            os.path.join(self.output_dir, "compression_ratios.csv"), 
            index=False
        )
        
        logger.info("Compression ratio test completed.")
    
    def run_relevance_test(self) -> None:
        """Test the relevance evaluation component."""
        logger.info("Starting relevance evaluation test...")
        
        # Create test segments with varying degrees of relevance
        current_context = """
        We've been discussing neural networks and their applications in computer vision.
        Specifically, we covered convolutional neural networks (CNNs) and how they use
        filters to detect features in images. We also touched on transfer learning as 
        a technique to leverage pre-trained models.
        """
        
        test_queries = [
            "Can you explain how CNNs work?",
            "What are some applications of transfer learning?",
            "How do neural networks compare to traditional computer vision techniques?"
        ]
        
        test_segments = [
            # High relevance
            MemorySegment(
                content="Convolutional Neural Networks (CNNs) are designed for processing data with grid-like topology, such as images. They use convolutional layers that apply filters to detect features like edges, textures, and patterns. Each filter slides across the input, creating a feature map highlighting where specific patterns occur.",
                id="high_rel_1",
                creation_time=time.time() - 3600,  # 1 hour ago
                importance=0.8
            ),
            # Medium relevance
            MemorySegment(
                content="Transfer learning allows you to leverage knowledge from pre-trained models on large datasets. Instead of training from scratch, you can use the learned feature representations and fine-tune the model on your specific task, which is particularly useful when you have limited data.",
                id="med_rel_1",
                creation_time=time.time() - 7200,  # 2 hours ago
                importance=0.6
            ),
            # Low relevance
            MemorySegment(
                content="Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward. It's been successfully applied to game playing, robotics, and autonomous driving.",
                id="low_rel_1",
                creation_time=time.time() - 1800,  # 30 minutes ago
                importance=0.4
            ),
            # Irrelevant
            MemorySegment(
                content="Climate change is leading to rising sea levels, more frequent extreme weather events, and changing precipitation patterns. Mitigation strategies include reducing carbon emissions, transitioning to renewable energy, and improving energy efficiency.",
                id="irrel_1",
                creation_time=time.time() - 300,  # 5 minutes ago
                importance=0.2
            )
        ]
        
        relevance_scores = []
        
        # Test each segment against the current context and queries
        for segment in test_segments:
            # Base relevance to context
            base_score = self.system.relevance_evaluator.calculate_importance(
                segment, current_context
            )
            
            # Relevance to each query
            query_scores = []
            for query in test_queries:
                score = self.system.relevance_evaluator.calculate_importance(
                    segment, current_context, query
                )
                query_scores.append(score)
            
            relevance_scores.append({
                "segment_id": segment.id,
                "content_sample": segment.content[:50] + "...",
                "age_hours": (time.time() - segment.creation_time) / 3600,
                "base_relevance": base_score,
                "query1_relevance": query_scores[0],
                "query2_relevance": query_scores[1],
                "query3_relevance": query_scores[2],
                "avg_query_relevance": sum(query_scores) / len(query_scores)
            })
        
        # Save results
        self.results["relevance_evaluation"] = relevance_scores
        pd.DataFrame(relevance_scores).to_csv(
            os.path.join(self.output_dir, "relevance_scores.csv"), 
            index=False
        )
        
        logger.info("Relevance evaluation test completed.")
    
    def test_retrieval_accuracy(self) -> None:
        """Test the accuracy of memory retrieval under different conditions."""
        logger.info("Starting retrieval accuracy test...")
        
        # Reset system for clean test
        self.system.reset_memory()
        
        # Create a sequence of related content with a specific fact
        target_facts = [
            "The AWS Lambda cold start time for Python functions is typically between 100-200ms.",
            "The melting point of tungsten is 3422°C, the highest of all metals.",
            "The Voyager 1 spacecraft is currently about 14 billion miles from Earth.",
            "Octopuses have three hearts: two pump blood through the gills, while the third pumps it through the body."
        ]
        
        # Generate conversation turns that include the target facts
        memory_segments = []
        for i, fact in enumerate(target_facts):
            # Add some context around the fact
            memory_segments.append({
                "content": f"Here's an interesting fact: {fact} This is something worth remembering.",
                "delay": i * 5  # Each fact is separated by more conversation turns
            })
        
        # Add distractors between facts
        topics = ["weather", "sports", "technology", "food", "travel", "movies", "music", "books"]
        for i in range(30):  # Add 30 distractor segments
            topic = random.choice(topics)
            memory_segments.append({
                "content": f"Let's talk about {topic}. There are many interesting aspects of {topic} worth discussing.",
                "delay": random.randint(0, 20)
            })
        
        # Sort based on insertion order
        memory_segments.sort(key=lambda x: x["delay"])
        
        # Insert memories into system
        for segment in memory_segments:
            self.system.memory_manager.add_to_hot_memory(segment["content"])
            
            # Simulate time passing and memory management
            if random.random() < 0.3:  # 30% chance of memory management
                self.system.memory_manager.manage_memory_tiers()
        
        # Force memory management to push some facts to warm/cold memory
        for _ in range(3):
            self.system.memory_manager.manage_memory_tiers()
        
        # Now test retrieval for each target fact
        retrieval_results = []
        for i, fact in enumerate(target_facts):
            # Construct query related to the fact
            if i == 0:
                query = "What was the cold start time for AWS Lambda functions?"
            elif i == 1:
                query = "What metal has the highest melting point?"
            elif i == 2:
                query = "How far is Voyager 1 from Earth?"
            else:
                query = "How many hearts does an octopus have?"
            
            # Process query
            response = self.system.process_input(query)
            
            # Check if the fact was retrieved correctly
            fact_retrieved = any(part in response for part in fact.split(". "))
            
            # Get current memory stats
            stats = self.system.get_memory_stats()
            
            retrieval_results.append({
                "target_fact": fact,
                "query": query,
                "fact_retrieved": fact_retrieved,
                "hot_segments": stats["hot_memory"]["segment_count"],
                "warm_segments": stats["warm_memory"]["segment_count"],
                "cold_segments": stats["cold_memory"]["segment_count"]
            })
        
        # Save results
        self.results["retrieval_accuracy"] = retrieval_results
        pd.DataFrame(retrieval_results).to_csv(
            os.path.join(self.output_dir, "retrieval_accuracy.csv"), 
            index=False
        )
        
        logger.info("Retrieval accuracy test completed.")
    
    def test_decompression_fidelity(self) -> None:
        """Test how well decompression preserves original meaning."""
        logger.info("Starting decompression fidelity test...")
        
        # Define test content
        test_content = [
            "The transformer architecture relies on self-attention mechanisms to capture dependencies between input tokens. Instead of processing data sequentially, transformers can consider all tokens simultaneously, making them highly parallelizable and effective for processing sequential data like text.",
            "Python's Global Interpreter Lock (GIL) prevents multiple threads from executing Python bytecodes at once. This means that in CPython, threads cannot execute Python code in parallel, though I/O operations can run concurrently. The GIL simplifies memory management but can limit CPU-bound multithreaded performance.",
            "A recurrent neural network (RNN) processes sequences by maintaining a hidden state that's updated at each time step. This allows the network to have 'memory' of previous inputs, making RNNs suitable for tasks like language modeling, speech recognition, and time series prediction."
        ]
        
        results = []
        
        for content in test_content:
            # Create original segment
            original = MemorySegment(
                content=content,
                id=f"test_decompress_{len(content)%100}",
                creation_time=time.time(),
                importance=0.5
            )
            
            # Step 1: Light compression
            lightly_compressed = self.system.compressor.compress(original, level=1)
            
            # Step 2: Heavy compression
            heavily_compressed = self.system.compressor.compress(original, level=2)
            
            # Step 3: Decompress both
            lightly_decompressed = self.system.retrieval_engine.decompress_segment(lightly_compressed)
            heavily_decompressed = self.system.retrieval_engine.decompress_segment(heavily_compressed)
            
            # Step 4: Extract entities from all versions
            original_entities = self.system.entity_extractor.extract_entities(original.content)
            light_entities = self.system.entity_extractor.extract_entities(lightly_decompressed.content)
            heavy_entities = self.system.entity_extractor.extract_entities(heavily_decompressed.content)
            
            # Calculate overlap metrics
            light_entity_preservation = len(set(light_entities).intersection(set(original_entities))) / len(original_entities) if original_entities else 1.0
            heavy_entity_preservation = len(set(heavy_entities).intersection(set(original_entities))) / len(original_entities) if original_entities else 1.0
            
            # Get semantic similarity
            light_semantic_overlap = self.system.relevance_evaluator._calculate_semantic_similarity(
                lightly_decompressed, original.content
            )
            heavy_semantic_overlap = self.system.relevance_evaluator._calculate_semantic_similarity(
                heavily_decompressed, original.content
            )
            
            results.append({
                "original_length": original.get_size(),
                "light_compressed_length": lightly_compressed.get_size(),
                "heavy_compressed_length": heavily_compressed.get_size(),
                "light_decompressed_length": lightly_decompressed.get_size(),
                "heavy_decompressed_length": heavily_decompressed.get_size(),
                "light_compression_ratio": original.get_size() / lightly_compressed.get_size() if lightly_compressed.get_size() > 0 else float('inf'),
                "heavy_compression_ratio": original.get_size() / heavily_compressed.get_size() if heavily_compressed.get_size() > 0 else float('inf'),
                "original_entities": len(original_entities),
                "light_entity_preservation": light_entity_preservation,
                "heavy_entity_preservation": heavy_entity_preservation,
                "light_semantic_overlap": light_semantic_overlap,
                "heavy_semantic_overlap": heavy_semantic_overlap
            })
        
        # Save results
        self.results["decompression_fidelity"] = results
        pd.DataFrame(results).to_csv(
            os.path.join(self.output_dir, "decompression_fidelity.csv"), 
            index=False
        )
        
        logger.info("Decompression fidelity test completed.")
    
    def generate_visualizations(self) -> None:
        """Generate visualizations from benchmark results."""
        logger.info("Generating visualizations...")
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set Seaborn style
        sns.set(style="whitegrid")
        plt.rcParams.update({'figure.figsize': (12, 8)})
        
        # 1. Memory utilization over time
        if self.results["memory_utilization"]:
            df = pd.DataFrame(self.results["memory_utilization"])
            
            plt.figure(figsize=(14, 8))
            plt.subplot(2, 1, 1)
            plt.plot(df['turn'], df['hot_size'], 'r-', label='Hot Memory')
            plt.plot(df['turn'], df['warm_size'], 'b-', label='Warm Memory')
            plt.plot(df['turn'], df['cold_size'], 'g-', label='Cold Memory')
            plt.axhline(y=self.hot_size, color='r', linestyle='--', label='Hot Capacity')
            plt.axhline(y=self.warm_size, color='b', linestyle='--', label='Warm Capacity')
            plt.axhline(y=self.cold_size, color='g', linestyle='--', label='Cold Capacity')
            plt.title('Memory Utilization Over Time')
            plt.xlabel('Conversation Turn')
            plt.ylabel('Memory Size (tokens)')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(df['turn'], df['hot_segments'], 'r-', label='Hot Segments')
            plt.plot(df['turn'], df['warm_segments'], 'b-', label='Warm Segments')
            plt.plot(df['turn'], df['cold_segments'], 'g-', label='Cold Segments')
            plt.title('Memory Segment Count Over Time')
            plt.xlabel('Conversation Turn')
            plt.ylabel('Number of Segments')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'memory_utilization.png'), dpi=300)
            plt.close()
        
        # 2. Compression ratios by content type
        if self.results["compression_ratios"]:
            df = pd.DataFrame(self.results["compression_ratios"])
            
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 1, 1)
            sns.barplot(x='content_type', y='l1_ratio', data=df, color='skyblue', label='Level 1')
            sns.barplot(x='content_type', y='l2_ratio', data=df, color='navy', label='Level 2')
            plt.title('Compression Ratios by Content Type')
            plt.xlabel('Content Type')
            plt.ylabel('Compression Ratio')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            sns.barplot(x='content_type', y='l1_entity_preservation', data=df, color='lightgreen', label='Level 1')
            sns.barplot(x='content_type', y='l2_entity_preservation', data=df, color='darkgreen', label='Level 2')
            plt.title('Entity Preservation by Content Type')
            plt.xlabel('Content Type')
            plt.ylabel('Entity Preservation Ratio')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'compression_ratios.png'), dpi=300)
            plt.close()
        
        # 3. Response times histogram
        if self.results["response_times"]:
            plt.figure(figsize=(10, 6))
            plt.hist(self.results["response_times"], bins=20, color='purple', alpha=0.7)
            plt.axvline(x=np.mean(self.results["response_times"]), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(self.results["response_times"]):.2f}s')
            plt.axvline(x=np.median(self.results["response_times"]), color='b', linestyle='--', 
                      label=f'Median: {np.median(self.results["response_times"]):.2f}s')
            plt.title('Response Time Distribution')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(os.path.join(plots_dir, 'response_times.png'), dpi=300)
            plt.close()
        
        # 4. Relevance scores by segment type
        if "relevance_evaluation" in self.results:
            df = pd.DataFrame(self.results["relevance_evaluation"])
            
            plt.figure(figsize=(12, 8))
            
            # Reshape data for plotting
            scores_df = df.melt(id_vars=["segment_id", "content_sample"], 
                                value_vars=["base_relevance", "query1_relevance", 
                                            "query2_relevance", "query3_relevance"],
                                var_name="Query Type", value_name="Relevance Score")
            