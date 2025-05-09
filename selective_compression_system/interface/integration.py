import logging
from typing import List, Dict, Any

class IntegrationLayer:
    def __init__(self, memory_manager, retrieval_engine, llm_interface):
        """Initialize the integration layer.
        
        Args:
            memory_manager: Memory manager instance
            retrieval_engine: Retrieval engine instance
            llm_interface: LLM interface instance
        """
        self.memory_manager = memory_manager
        self.retrieval_engine = retrieval_engine
        self.llm_interface = llm_interface
        logging.info("Integration layer initialized")

    def process_user_input(self, user_input):
        """Process user input and generate response.
        
        Args:
            user_input (str): Input from the user
            
        Returns:
            str: Generated response
        """
        try:
            # 1. Add user input to hot memory
            self.memory_manager.add_to_hot_memory(user_input)
            logging.debug("Added user input to hot memory")
            
            # 2. Get current context from hot memory
            current_context = self._get_current_context()
            
            # 3. Identify relevant compressed memories
            relevant_segments = self.retrieval_engine.identify_relevant_segments(
                current_context, user_input
            )
            logging.debug(f"Found {len(relevant_segments)} relevant memory segments")
            
            # 4. Decompress relevant segments
            decompressed_segments = []
            for segment in relevant_segments:
                try:
                    decompressed = self.retrieval_engine.decompress_segment(segment)
                    decompressed_segments.append(decompressed)
                except Exception as e:
                    logging.error(f"Error decompressing segment {segment.id}: {str(e)}")
            
            # 5. Construct full context for LLM
            full_context = self._construct_context(current_context, decompressed_segments)
            
            # 6. Generate response
            llm_response = self.llm_interface.generate_response(full_context, user_input)
            
            # 7. Add response to hot memory
            self.memory_manager.add_to_hot_memory(llm_response)
            
            # 8. Run memory maintenance
            self.memory_manager.manage_memory_tiers()
            
            return llm_response
            
        except Exception as e:
            logging.error(f"Error processing user input: {str(e)}")
            return f"I'm having trouble processing your request. {str(e)}"
            
    def _get_current_context(self):
        """Get current context from hot memory.
        
        Returns:
            str: Concatenated content from hot memory segments
        """
        try:
            hot_segments = self.memory_manager.hot_memory.get_all_segments()
            
            # Sort by creation time to maintain conversation flow
            hot_segments.sort(key=lambda x: x.creation_time)
            
            # Concatenate content
            if hot_segments:
                return "\n".join([segment.content for segment in hot_segments])
            else:
                return ""
        except Exception as e:
            logging.error(f"Error getting current context: {str(e)}")
            return ""
            
    def _construct_context(self, current_context, decompressed_segments):
        """Combine hot memory with decompressed relevant segments.
        
        Args:
            current_context (str): Current hot memory context
            decompressed_segments (List): List of decompressed memory segments
            
        Returns:
            str: Combined context for LLM
        """
        try:
            # Start with current hot memory
            context_parts = []
            if current_context.strip():
                context_parts.append(current_context)
            
            # Add decompressed segments if any
            if decompressed_segments:
                context_parts.append("\n\n--- Relevant Previous Context ---\n")
                for segment in decompressed_segments:
                    context_parts.append(segment.content)
                    
            return "\n".join(context_parts)
        except Exception as e:
            logging.error(f"Error constructing context: {str(e)}")
            # Return original context as fallback
            return current_context