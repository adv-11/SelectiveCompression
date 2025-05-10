import argparse
import time
import json
import os
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the system components
from core.system import SelectiveCompressionSystem
from core.memory import MemorySegment

class CompressionBenchmark:
    """Class to benchmark the selective compression memory system"""
    
    def __init__(self, model_name="gpt-4o-mini", hot_size=4000, warm_size=16000, cold_size=64000, 
                 output_dir="benchmark_results", api_key=None):
        """Initialize the benchmark.
        
        Args:
            model_name (str): Name of the LLM model to use
            hot_size (int): Size of hot memory in tokens
            warm_size (int): Size of warm memory in tokens
            cold_size (int): Size of cold memory in tokens
            output_dir (str): Directory to save results
            api_key (str, optional): API key for LLM service
        """
        self.model_name = model_name
        self.hot_size = hot_size
        self.warm_size = warm_size
        self.cold_size = cold_size
        self.output_dir = output_dir
        self.api_key = api_key
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize system
        self.system = SelectiveCompressionSystem(
            model_name=model_name,
            hot_size=hot_size,
            warm_size=warm_size,
            cold_size=cold_size,
            api_key=api_key
        )
        
        # Metrics to track
        self.metrics = {
            "response_times": [],
            "compression_ratios": [],
            "memory_usage": [],
            "retrieval_count": [],
            "relevance_scores": [],
            "compression_times": [],
            "decompression_times": [],
            "tier_transitions": {
                "hot_to_warm": 0,
                "warm_to_cold": 0,
                "cold_to_evict": 0
            },
            "tier_sizes_over_time": {
                "hot": [],
                "warm": [],
                "cold": []
            },
            "conversation_length": 0
        }
        
        logger.info(f"Initialized benchmark with model {model_name}")
        
    def load_conversation_dataset(self, file_path):
        """Load a conversation dataset from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            list: List of conversation turns
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded conversation dataset with {len(data)} turns")
            return data
        except Exception as e:
            logger.error(f"Error loading conversation dataset: {str(e)}")
            # Create simple fallback dataset
            return [
                {"user": "Hello, how are you today?"},
                {"user": "Can you tell me about memory systems in AI?"},
                {"user": "What are the challenges with long-term memory in LLMs?"},
                {"user": "How does selective compression help with memory?"},
                {"user": "Can you explain the difference between episodic and semantic memory?"}
            ]
            
    def create_synthetic_dataset(self, size=100, topic_clusters=5, noise_factor=0.2):
        """Create a synthetic conversation dataset with topic clusters.
        
        Args:
            size (int): Number of conversation turns
            topic_clusters (int): Number of distinct topics
            noise_factor (float): Probability of random unrelated messages
            
        Returns:
            list: List of conversation turns
        """
        topics = [
            "Artificial Intelligence and Machine Learning",
            "Memory Systems in Cognitive Science",
            "Software Engineering and Architecture",
            "Natural Language Processing",
            "Ethics and Bias in AI",
            "Climate Change and Environmental Science",
            "Biotechnology and Medicine",
            "Quantum Computing",
            "Space Exploration and Astronomy",
            "Robotics and Automation"
        ]
        
        # Select a subset of topics
        selected_topics = random.sample(topics, min(topic_clusters, len(topics)))
        
        # Generate questions for each topic
        topic_questions = {}
        for topic in selected_topics:
            questions = self._generate_topic_questions(topic, size // topic_clusters)
            topic_questions[topic] = questions
            
        # Create dataset with interleaved topics
        dataset = []
        topic_order = []
        
        # Create a sequence that interleaves topics with some repetition
        for i in range(size):
            # Either continue current topic or switch
            if i > 0 and random.random() < 0.7 and topic_order[-1] in selected_topics:
                topic = topic_order[-1]  # Stay on same topic
            else:
                # Switch topics
                topic = random.choice(selected_topics)
            
            topic_order.append(topic)
            
        # Add noise to the topic sequence
        for i in range(size):
            if random.random() < noise_factor:
                topic_order[i] = "Random"
        
        # Generate turns from topics
        for i, topic in enumerate(topic_order):
            if topic == "Random":
                # Random unrelated question
                dataset.append({"user": f"Random question {i}: {self._generate_random_question()}"})
            else:
                # Question from the topic
                questions = topic_questions[topic]
                question_index = i % len(questions)
                dataset.append({"user": questions[question_index]})
                
        logger.info(f"Created synthetic dataset with {len(dataset)} turns across {topic_clusters} topics")
        return dataset
    
    def _generate_topic_questions(self, topic, count):
        """Generate questions related to a topic.
        
        Args:
            topic (str): Topic to generate questions for
            count (int): Number of questions to generate
            
        Returns:
            list: List of questions
        """
        # Template questions by topic
        templates = {
            "Artificial Intelligence and Machine Learning": [
                "What are the key differences between {x} and {y} in machine learning?",
                "How does {x} improve model performance in AI systems?",
                "Can you explain the concept of {x} in the context of deep learning?",
                "What are the limitations of using {x} for {y} tasks?",
                "How is {x} implemented in practical AI applications?"
            ],
            "Memory Systems in Cognitive Science": [
                "How does {x} memory work in the human brain?",
                "What's the relationship between {x} and {y} in memory formation?",
                "Can you explain the role of {x} in memory retrieval?",
                "How do {x} memory systems differ from {y} systems?",
                "What are the key mechanisms of {x} in memory consolidation?"
            ],
            "Software Engineering and Architecture": [
                "What are best practices for implementing {x} in software architecture?",
                "How does {x} compare to {y} in terms of scalability?",
                "Can you explain the principles of {x} in modern software design?",
                "What are the challenges of maintaining {x} in large codebases?",
                "How does {x} architecture handle {y} problems?"
            ],
            "Natural Language Processing": [
                "How do {x} models handle {y} in text processing?",
                "What are the limitations of {x} for understanding context?",
                "Can you explain how {x} improves {y} in NLP systems?",
                "What's the difference between {x} and {y} approaches to NLP?",
                "How does {x} help with {y} in language understanding?"
            ],
            "Ethics and Bias in AI": [
                "How does {x} bias manifest in AI systems?",
                "What methods are effective for reducing {x} in ML models?",
                "Can you explain the ethical implications of {x} in AI development?",
                "How do {x} and {y} biases interact in decision systems?",
                "What responsibility do developers have regarding {x} issues?"
            ],
            "Climate Change and Environmental Science": [
                "How does {x} contribute to climate change?",
                "What are the effects of {x} on {y} ecosystems?",
                "Can you explain the relationship between {x} and {y} in environmental systems?",
                "What are the most promising solutions for addressing {x}?",
                "How does {x} impact global {y} patterns?"
            ],
            "Biotechnology and Medicine": [
                "How is {x} technology applied in modern medicine?",
                "What are the ethical considerations of using {x} for {y}?",
                "Can you explain how {x} works at the molecular level?",
                "What are the limitations of current {x} therapies?",
                "How does {x} compare to {y} in treating {z} conditions?"
            ],
            "Quantum Computing": [
                "How does {x} work in quantum computing systems?",
                "What are the challenges of implementing {x} in quantum algorithms?",
                "Can you explain the principle of {x} in quantum mechanics?",
                "How does {x} differ between classical and quantum computing?",
                "What are the practical applications of {x} in quantum technology?"
            ],
            "Space Exploration and Astronomy": [
                "What have we learned about {x} from recent space missions?",
                "How does {x} affect {y} in space environments?",
                "Can you explain the phenomenon of {x} in astronomical terms?",
                "What are the challenges of studying {x} with current technology?",
                "How does {x} relate to theories about {y}?"
            ],
            "Robotics and Automation": [
                "How are {x} sensors used in modern robotics?",
                "What are the challenges of implementing {x} in automated systems?",
                "Can you explain how {x} algorithms help robots navigate?",
                "How does {x} technology compare to {y} for automation tasks?",
                "What ethical considerations arise from {x} in robotics?"
            ]
        }
        
        # Topic-specific fillers
        fillers = {
            "Artificial Intelligence and Machine Learning": {
                "x": ["supervised learning", "unsupervised learning", "reinforcement learning", 
                      "neural networks", "decision trees", "support vector machines", "backpropagation",
                      "gradient descent", "transfer learning", "attention mechanisms"],
                "y": ["classification", "regression", "clustering", "dimensionality reduction",
                      "natural language processing", "computer vision", "speech recognition"]
            },
            "Memory Systems in Cognitive Science": {
                "x": ["episodic", "semantic", "procedural", "working", "short-term", "long-term",
                      "autobiographical", "implicit", "explicit", "declarative"],
                "y": ["encoding", "storage", "retrieval", "recall", "recognition", "forgetting",
                      "consolidation", "interference"]
            },
            "Software Engineering and Architecture": {
                "x": ["microservices", "monolithic", "serverless", "event-driven", "layered",
                      "object-oriented", "functional", "reactive", "MVC", "MVVM"],
                "y": ["scalability", "maintainability", "testability", "performance", "security",
                      "reliability", "availability", "fault tolerance"]
            },
            "Natural Language Processing": {
                "x": ["transformer", "BERT", "GPT", "RNN", "LSTM", "word embeddings", "tokenization",
                      "attention", "fine-tuning", "zero-shot learning"],
                "y": ["sentiment analysis", "named entity recognition", "text classification",
                      "machine translation", "summarization", "question answering"]
            },
            "Ethics and Bias in AI": {
                "x": ["algorithmic", "data", "selection", "confirmation", "representation", "evaluation",
                      "historical", "societal", "deployment", "feedback loop"],
                "y": ["racial", "gender", "socioeconomic", "cultural", "linguistic", "geographical"]
            },
            "Climate Change and Environmental Science": {
                "x": ["carbon emissions", "deforestation", "ocean acidification", "methane release",
                      "greenhouse gases", "industrialization", "urbanization", "agriculture"],
                "y": ["marine", "forest", "arctic", "tropical", "freshwater", "desert", "grassland"]
            },
            "Biotechnology and Medicine": {
                "x": ["CRISPR", "gene therapy", "stem cell", "immunotherapy", "mRNA", "antibody",
                      "genomic", "proteomic", "bioinformatic", "synthetic biology"],
                "y": ["cancer", "genetic disorders", "infectious diseases", "autoimmune conditions",
                      "neurological disorders", "aging", "regenerative medicine"],
                "z": ["chronic", "acute", "inherited", "acquired", "degenerative", "infectious"]
            },
            "Quantum Computing": {
                "x": ["superposition", "entanglement", "quantum gates", "qubits", "quantum annealing",
                      "quantum error correction", "quantum supremacy", "quantum tunneling"],
                "y": ["factorization", "optimization", "simulation", "cryptography", "machine learning"]
            },
            "Space Exploration and Astronomy": {
                "x": ["black holes", "exoplanets", "dark matter", "dark energy", "gravity waves",
                      "cosmic microwave background", "solar winds", "neutron stars"],
                "y": ["galaxy formation", "planetary systems", "stellar evolution", "cosmic expansion",
                      "habitability", "interstellar travel", "space colonization"]
            },
            "Robotics and Automation": {
                "x": ["computer vision", "path planning", "reinforcement learning", "SLAM", "haptic",
                      "force feedback", "robotic arms", "autonomous navigation"],
                "y": ["industrial", "medical", "domestic", "search and rescue", "agricultural",
                      "underwater", "space exploration", "military"]
            }
        }
        
        # Generate questions
        questions = []
        if topic in templates and topic in fillers:
            template_list = templates[topic]
            filler_dict = fillers[topic]
            
            for _ in range(count):
                # Select random template
                template = random.choice(template_list)
                
                # Fill in placeholders
                question = template
                for key in filler_dict:
                    if key in question:
                        value = random.choice(filler_dict[key])
                        question = question.replace(f"{{{key}}}", value)
                
                questions.append(question)
        else:
            # Default questions if topic not found
            for i in range(count):
                questions.append(f"Question {i+1} about {topic}")
                
        return questions
        
    def _generate_random_question(self):
        """Generate a random unrelated question.
        
        Returns:
            str: Random question
        """
        random_questions = [
            "What's your favorite color?",
            "Do you have any recommendations for good books?",
            "How do I make a good pasta sauce?",
            "Can you tell me a joke?",
            "What's the weather like today?",
            "How tall is the Empire State Building?",
            "Who won the World Cup in 2018?",
            "What's the capital of Australia?",
            "How do I change a tire on my car?",
            "What's the airspeed velocity of an unladen swallow?",
            "Can you recommend a good movie to watch?",
            "What's the difference between alligators and crocodiles?",
            "How do I get red wine stains out of carpet?",
            "What's the best way to learn a new language?",
            "How many planets are in our solar system?"
        ]
        return random.choice(random_questions)
    
    def run_benchmark(self, conversation_data, track_metrics=True):
        """Run benchmark on the conversation dataset.
        
        Args:
            conversation_data (list): List of conversation turns
            track_metrics (bool): Whether to track metrics
            
        Returns:
            dict: Benchmark results
        """
        logger.info(f"Starting benchmark with {len(conversation_data)} conversation turns")
        
        # Reset system
        self.system.reset_memory()
        
        # Reset metrics
        if track_metrics:
            self.metrics = {
                "response_times": [],
                "compression_ratios": [],
                "memory_usage": [],
                "retrieval_count": [],
                "relevance_scores": [],
                "compression_times": [],
                "decompression_times": [],
                "tier_transitions": {
                    "hot_to_warm": 0,
                    "warm_to_cold": 0,
                    "cold_to_evict": 0
                },
                "tier_sizes_over_time": {
                    "hot": [],
                    "warm": [],
                    "cold": []
                },
                "conversation_length": 0
            }
        
        # Process each conversation turn
        for i, turn in enumerate(conversation_data):
            logger.info(f"Processing turn {i+1}/{len(conversation_data)}")
            
            user_input = turn.get("user", "")
            if not user_input:
                continue
                
            # Record initial memory state
            if track_metrics:
                initial_stats = self.system.get_memory_stats()
                initial_hot_count = len(self.system.memory_manager.hot_memory.segments)
                initial_warm_count = len(self.system.memory_manager.warm_memory.segments)
                initial_cold_count = len(self.system.memory_manager.cold_memory.segments)
                
            # Process user input and time it
            start_time = time.time()
            response = self.system.process_input(user_input)
            end_time = time.time()
            response_time = end_time - start_time
            
            # Record post-processing memory state
            if track_metrics:
                # Update memory usage stats
                stats = self.system.get_memory_stats()
                self.metrics["memory_usage"].append({
                    "turn": i,
                    "hot": stats["hot_memory"]["size"],
                    "warm": stats["warm_memory"]["size"],
                    "cold": stats["cold_memory"]["size"],
                    "total": stats["total"]["size"]
                })
                
                # Count tier transitions
                final_hot_count = len(self.system.memory_manager.hot_memory.segments)
                final_warm_count = len(self.system.memory_manager.warm_memory.segments)
                final_cold_count = len(self.system.memory_manager.cold_memory.segments)
                
                # Track tier transitions
                hot_to_warm = max(0, initial_hot_count - final_hot_count + 2)  # +2 for user input and response
                warm_to_cold = max(0, initial_warm_count - final_warm_count + hot_to_warm)
                cold_to_evict = max(0, initial_cold_count - final_cold_count + warm_to_cold)
                
                self.metrics["tier_transitions"]["hot_to_warm"] += hot_to_warm
                self.metrics["tier_transitions"]["warm_to_cold"] += warm_to_cold
                self.metrics["tier_transitions"]["cold_to_evict"] += cold_to_evict
                
                # Track response time
                self.metrics["response_times"].append(response_time)
                
                # Track tier sizes
                self.metrics["tier_sizes_over_time"]["hot"].append(stats["hot_memory"]["size"])
                self.metrics["tier_sizes_over_time"]["warm"].append(stats["warm_memory"]["size"])
                self.metrics["tier_sizes_over_time"]["cold"].append(stats["cold_memory"]["size"])
                
                # Estimate compression ratios for any segments that moved tiers
                for segment_id in self.system.memory_manager.warm_memory.segments:
                    segment = self.system.memory_manager.warm_memory.segments[segment_id]
                    if hasattr(segment, 'metadata') and 'compression_ratio' in segment.metadata:
                        self.metrics["compression_ratios"].append(segment.metadata['compression_ratio'])
                        
                self.metrics["conversation_length"] += 1
                
            # Optional: sleep to avoid rate limiting
            time.sleep(0.5)
            
        logger.info(f"Benchmark completed with {self.metrics['conversation_length']} turns")
        return self.metrics
    
    def evaluate_memory_retrieval(self, test_queries, reference_context, relevance_threshold=0.6):
        """Evaluate memory retrieval performance.
        
        Args:
            test_queries (list): List of test queries
            reference_context (str): Reference context to compare against
            relevance_threshold (float): Threshold for relevant retrieval
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating memory retrieval with {len(test_queries)} test queries")
        
        results = {
            "retrieval_accuracy": [],
            "retrieval_relevance": [],
            "retrieval_time": []
        }
        
        for i, query in enumerate(test_queries):
            logger.info(f"Processing test query {i+1}/{len(test_queries)}")
            
            # Process query and time it
            start_time = time.time()
            # Get current context from hot memory
            current_context = self.system.get_hot_memory_contents()
            
            # Find relevant memory segments
            relevant_segments = self.system.retrieval_engine.identify_relevant_segments(
                current_context, query
            )
            end_time = time.time()
            retrieval_time = end_time - start_time
            
            # Calculate relevance to reference context
            relevance_scores = []
            for segment in relevant_segments:
                relevance = self.system.relevance_evaluator.calculate_importance(
                    segment, reference_context, query
                )
                relevance_scores.append(relevance)
            
            # Calculate metrics
            avg_relevance = sum(relevance_scores) / max(1, len(relevance_scores))
            accuracy = len([r for r in relevance_scores if r > relevance_threshold]) / max(1, len(relevance_scores))
            
            results["retrieval_accuracy"].append(accuracy)
            results["retrieval_relevance"].append(avg_relevance)
            results["retrieval_time"].append(retrieval_time)
            
        # Calculate summary statistics
        results["avg_accuracy"] = sum(results["retrieval_accuracy"]) / max(1, len(results["retrieval_accuracy"]))
        results["avg_relevance"] = sum(results["retrieval_relevance"]) / max(1, len(results["retrieval_relevance"]))
        results["avg_retrieval_time"] = sum(results["retrieval_time"]) / max(1, len(results["retrieval_time"]))
        
        logger.info(f"Memory retrieval evaluation complete")
        return results
    
    def evaluate_compression_quality(self, sample_size=10):
        """Evaluate the quality of compression.
        
        Args:
            sample_size (int): Number of segments to evaluate
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating compression quality with sample size {sample_size}")
        
        results = {
            "compression_ratios": [],
            "semantic_similarity": [],
            "entity_preservation": []
        }
        
        # Sample segments from warm memory
        warm_segments = list(self.system.memory_manager.warm_memory.segments.values())
        if len(warm_segments) == 0:
            logger.warning("No segments in warm memory for compression evaluation")
            return results
            
        samples = random.sample(warm_segments, min(sample_size, len(warm_segments)))
        
        for i, segment in enumerate(samples):
            logger.info(f"Evaluating compression for segment {i+1}/{len(samples)}")
            
            # Get compression ratio
            if hasattr(segment, 'metadata') and 'compression_ratio' in segment.metadata:
                compression_ratio = segment.metadata['compression_ratio']
                results["compression_ratios"].append(compression_ratio)
                
                # Get original entities if available
                if 'original_entities' in segment.metadata:
                    original_entities = segment.metadata['original_entities']
                    
                    # Extract entities from compressed content
                    compressed_entities = self.system.entity_extractor.extract_entities(segment.content)
                    
                    # Calculate entity preservation
                    if original_entities:
                        preserved = len(set(compressed_entities).intersection(set(original_entities)))
                        preservation_rate = preserved / len(original_entities)
                        results["entity_preservation"].append(preservation_rate)
                        
                # Decompress segment
                try:
                    decompressed = self.system.retrieval_engine.decompress_segment(segment)
                    
                    # Calculate semantic similarity between original and decompressed
                    if hasattr(segment, 'metadata') and 'original_length' in segment.metadata:
                        # We don't have the original content, so we'll use semantic similarity
                        # with the decompressed content as a proxy
                        semantic_similarity = self.system.relevance_evaluator._calculate_semantic_similarity(
                            segment, decompressed.content
                        )
                        results["semantic_similarity"].append(semantic_similarity)
                except Exception as e:
                    logger.error(f"Error decompressing segment: {str(e)}")
        
        # Calculate summary statistics
        results["avg_compression_ratio"] = sum(results["compression_ratios"]) / max(1, len(results["compression_ratios"]))
        results["avg_semantic_similarity"] = sum(results["semantic_similarity"]) / max(1, len(results["semantic_similarity"]))
        results["avg_entity_preservation"] = sum(results["entity_preservation"]) / max(1, len(results["entity_preservation"]))
        
        logger.info(f"Compression quality evaluation complete")
        return results
    
    def plot_memory_usage(self, save_path=None):
        """Plot memory usage over time.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.metrics["memory_usage"]:
            logger.warning("No memory usage data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        
        turns = [entry["turn"] for entry in self.metrics["memory_usage"]]
        hot = [entry["hot"] for entry in self.metrics["memory_usage"]]
        warm = [entry["warm"] for entry in self.metrics["memory_usage"]]
        cold = [entry["cold"] for entry in self.metrics["memory_usage"]]
        total = [entry["total"] for entry in self.metrics["memory_usage"]]
        
        plt.plot(turns, hot, 'r-', label='Hot Memory')
        plt.plot(turns, warm, 'g-', label='Warm Memory')
        plt.plot(turns, cold, 'b-', label='Cold Memory')
        plt.plot(turns, total, 'k--', label='Total Memory')
        
        plt.title('Memory Usage Over Time')
        plt.xlabel('Conversation Turn')
        plt.ylabel('Memory Size (tokens)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis to show only integer values
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved memory usage plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_tier_transitions(self, save_path=None):
        """Plot memory tier transitions.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.metrics["tier_transitions"]:
            logger.warning("No tier transitions data to plot")
            return
            
        plt.figure(figsize=(8, 6))
        
        transitions = self.metrics["tier_transitions"]
        labels = ["Hot → Warm", "Warm → Cold", "Cold → Evicted"]
        values = [transitions["hot_to_warm"], transitions["warm_to_cold"], transitions["cold_to_evict"]]
        
        plt.bar(labels, values, color=['#ff9999', '#66b3ff', '#99ff99'])
        
        plt.title('Memory Tier Transitions')
        plt.xlabel('Transition Type')
        plt.ylabel('Number of Transitions')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(values):
            plt.text(i, v + 0.5, str(v), ha='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved tier transitions plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_response_times(self, save_path=None):
        """Plot response times over conversation turns.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.metrics["response_times"]:
            logger.warning("No response time data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        
        turns = list(range(1, len(self.metrics["response_times"]) + 1))
        times = self.metrics["response_times"]
        
        plt.plot(turns, times, 'b-', marker='o', markersize=4)
        
        # Add trend line
        z = np.polyfit(turns, times, 1)
        p = np.poly1d(z)
        plt.plot(turns, p(turns), "r--", alpha=0.7, label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")
        
        plt.title('Response Time Over Conversation Length')
        plt.xlabel('Conversation Turn')
        plt.ylabel('Response Time (seconds)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis to show only integer values
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved response times plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    def plot_compression_ratios(self, save_path=None):
        """Plot compression ratios.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.metrics["compression_ratios"]:
            logger.warning("No compression ratio data to plot")
            return
            
        plt.figure(figsize=(8, 6))
        
        # Create histogram of compression ratios
        plt.hist(self.metrics["compression_ratios"], bins=10, color='skyblue', edgecolor='black')
        
        plt.title('Distribution of Compression Ratios')
        plt.xlabel('Compression Ratio')
        plt.ylabel('Frequency')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add mean line
        mean_ratio = sum(self.metrics["compression_ratios"]) / len(self.metrics["compression_ratios"])
        plt.axvline(mean_ratio, color='r', linestyle='--', label=f'Mean: {mean_ratio:.2f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved compression ratios plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def save_results(self):
        """Save benchmark results to files.
        
        Returns:
            str: Path to results directory
        """
        # Create results directory
        results_dir = os.path.join(self.output_dir, f"benchmark_{self.timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = os.path.join(results_dir, "metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, (list, dict)):
                    serializable_metrics[key] = value
                elif isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                else:
                    serializable_metrics[key] = value
                    
            json.dump(serializable_metrics, f, indent=4)
        
        # Generate and save plots
        self.plot_memory_usage(save_path=os.path.join(results_dir, "memory_usage.png"))
        self.plot_tier_transitions(save_path=os.path.join(results_dir, "tier_transitions.png"))
        self.plot_response_times(save_path=os.path.join(results_dir, "response_times.png"))
        self.plot_compression_ratios(save_path=os.path.join(results_dir, "compression_ratios.png"))
        
        # Save configuration
        config = {
            "model_name": self.model_name,
            "hot_size": self.hot_size,
            "warm_size": self.warm_size,
            "cold_size": self.cold_size,
            "timestamp": self.timestamp
        }
        
        config_file = os.path.join(results_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
            
        logger.info(f"Saved benchmark results to {results_dir}")
        return results_dir
    
    def run_full_benchmark(self, conversation_data=None, dataset_path=None, synthetic=False, 
                          synthetic_size=100, synthetic_topics=5):
        """Run a complete benchmark with all evaluations.
        
        Args:
            conversation_data (list, optional): List of conversation turns
            dataset_path (str, optional): Path to conversation dataset
            synthetic (bool): Whether to use synthetic data
            synthetic_size (int): Size of synthetic dataset
            synthetic_topics (int): Number of topics in synthetic dataset
            
        Returns:
            dict: Benchmark results
        """
        # Load conversation data
        if conversation_data is None:
            if synthetic:
                conversation_data = self.create_synthetic_dataset(
                    size=synthetic_size, topic_clusters=synthetic_topics
                )
            elif dataset_path:
                conversation_data = self.load_conversation_dataset(dataset_path)
            else:
                # Create a small synthetic dataset
                conversation_data = self.create_synthetic_dataset(size=20, topic_clusters=3)
        
        # Run main benchmark
        metrics = self.run_benchmark(conversation_data)
        
        # Create test queries for retrieval evaluation
        if len(conversation_data) > 10:
            # Use some of the input as test queries
            test_queries = [turn["user"] for turn in random.sample(conversation_data, 5)]
            # Use hot memory as reference context
            reference_context = self.system.get_hot_memory_contents()
            
            # Evaluate memory retrieval
            retrieval_results = self.evaluate_memory_retrieval(test_queries, reference_context)
            metrics["retrieval_evaluation"] = retrieval_results
        
        # Evaluate compression quality
        compression_results = self.evaluate_compression_quality()
        metrics["compression_evaluation"] = compression_results
        
        # Save results
        results_dir = self.save_results()
        
        logger.info(f"Full benchmark completed. Results saved to {results_dir}")
        return metrics

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Run memory system benchmark")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--hot-size", type=int, default=4000, help="Hot memory size in tokens")
    parser.add_argument("--warm-size", type=int, default=16000, help="Warm memory size in tokens")
    parser.add_argument("--cold-size", type=int, default=64000, help="Cold memory size in tokens")
    parser.add_argument("--dataset", help="Path to conversation dataset JSON file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    parser.add_argument("--synthetic-size", type=int, default=100, help="Size of synthetic dataset")
    parser.add_argument("--synthetic-topics", type=int, default=5, help="Number of topics in synthetic dataset")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory to save results")
    parser.add_argument("--api-key", help="API key for LLM service")
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = CompressionBenchmark(
        model_name=args.model,
        hot_size=args.hot_size,
        warm_size=args.warm_size,
        cold_size=args.cold_size,
        output_dir=args.output_dir,
        api_key=args.api_key
    )
    
    # Run benchmark
    benchmark.run_full_benchmark(
        dataset_path=args.dataset,
        synthetic=args.synthetic,
        synthetic_size=args.synthetic_size,
        synthetic_topics=args.synthetic_topics
    )

if __name__ == "__main__":
    main()