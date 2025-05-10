import openai
import os
import logging
from typing import List, Dict, Any, Optional
import time
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

class LLMInterface:
    def __init__(self, model_name, api_key=None):
        """Initialize LLM interface with model name and API key.
        
        Args:
            model_name (str): Name of the OpenAI model to use
            api_key (str, optional): OpenAI API key. If None, uses environment variable.
        """
        self.model_name = model_name
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            logging.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        
        # Configure OpenAI client
        try:
            openai.api_key = self.api_key
        except Exception as e:
            logging.error(f"Error configuring OpenAI client: {str(e)}")
            raise

    def generate(self, prompt: str) -> str:
        """Generate text from prompt using the OpenAI API.
        
        Args:
            prompt (str): Input prompt for text generation
            
        Returns:
            str: Generated text response
        """
        try:
            # Rate limiting protection
            time.sleep(0.5)
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error in LLM generation: {str(e)}")
            # Fallback response in case of API failure
            return f"Error generating response: {str(e)[:100]}..."

    def generate_response(self, context: str, user_input: str) -> str:
        """Generate a response to user input given context.
        
        Args:
            context (str): Conversation context
            user_input (str): User's input/query
            
        Returns:
            str: Generated response
        """
        # Construct prompt including context and user input
        prompt = self._construct_prompt(context, user_input)
        
        # Generate response
        return self.generate(prompt)

    def embed_text(self, text: str) -> List[float]:
        """Get embedding vector for input text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            # Rate limiting protection
            time.sleep(0.2)
            
            # Use text-embedding model
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text[:8000]  # Limit text length to model constraints
            )
            
            return response["data"][0]["embedding"]
        except Exception as e:
            logging.error(f"Error in text embedding: {str(e)}")
            # Fallback to simple random vector in case of API failure
            import random
            return [random.random() for _ in range(1536)]  # Ada embeddings are 1536 dimensions

    def _construct_prompt(self, context: str, user_input: str) -> str:
        """Construct prompt with context and user input for the LLM.
        
        Args:
            context (str): Conversation context
            user_input (str): User's input/query
            
        Returns:
            str: Constructed prompt
        """
        return f"""
        You are an intelligent assistant with memory capabilities. 
        Use the following context from previous conversations when forming your response.
        
        Context:
        {context}
        
        User: {user_input}
        
        Assistant:
        """