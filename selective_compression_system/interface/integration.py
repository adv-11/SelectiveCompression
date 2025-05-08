
class IntegrationLayer:

    def init(self, memory_manager, retrieval_engine, llm_interface):

        self.memory_manager = memory_manager
        self.retrieval_engine = retrieval_engine
        self.llm_interface = llm_interface

        def process_user_input(self, user_input):

            """Process user input and generate response"""
            # 1. Add user input to hot memory
            self.memory_manager.add_to_hot_memory(user_input)
            
            # 2. Get current context from hot memory
            current_context = self._get_current_context()
            
            # 3. Identify relevant compressed memories
            relevant_segments = self.retrieval_engine.identify_relevant_segments(
                current_context, user_input
            )
            
            # 4. Decompress relevant segments
            decompressed_segments = []
            for segment in relevant_segments:
                decompressed = self.retrieval_engine.decompress_segment(segment)
                decompressed_segments.append(decompressed)
                
            # 5. Construct full context for LLM
            full_context = self._construct_context(current_context, decompressed_segments)
            
            # 6. Generate response
            llm_response = self.llm_interface.generate_response(full_context, user_input)
            
            # 7. Add response to hot memory
            self.memory_manager.add_to_hot_memory(llm_response)
            
            # 8. Run memory maintenance
            self.memory_manager.manage_memory_tiers()
            
            return llm_response
            
        def _get_current_context(self):

            """Get current context from hot memory"""
            hot_segments = self.memory_manager.hot_memory.get_all_segments()
            
            # Sort by creation time to maintain conversation flow
            hot_segments.sort(key=lambda x: x.creation_time)
            
            # Concatenate content
            return "\n".join([segment.content for segment in hot_segments])
            
        def _construct_context(self, current_context, decompressed_segments):

            """Combine hot memory with decompressed relevant segments"""
            # Start with current hot memory
            context_parts = [current_context]
            
            # Add decompressed segments if any
            if decompressed_segments:
                context_parts.append("\n\n--- Relevant Previous Context ---\n")
                for segment in decompressed_segments:
                    context_parts.append(segment.content)
                    
            return "\n".join(context_parts)