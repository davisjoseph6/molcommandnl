import os
from typing import List, Optional, Dict, Any, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms.ollama import Ollama

from utils import load_config
# Load env config
script_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(script_dir + '/../config/env')

QUERYLLM = config.get('QUERYLLM')

if (OLLAMA_HOST := config.get('OLLAMA_HOST')):
    print(f"Setting OLLAMA_HOST to {OLLAMA_HOST}")
    os.environ["OLLAMA_HOST"] = OLLAMA_HOST



def load_prompt(name, base_path, return_dict=False):
    """
    Load system and user prompts for a given task.

    Args:
        name (str): Logical name of the task (e.g., 'entity_context').
        base_path (str): Directory where prompts are stored.
        return_dict (bool): If True, returns a dict { 'system': ..., 'user': ... },
                            otherwise returns a tuple (system_prompt, user_prompt).

    Returns:
        tuple or dict: The system and user prompts.
    """
    system_file = os.path.join(base_path, f"system_prompt_{name}.txt")
    user_file = os.path.join(base_path, f"user_prompt_{name}.txt")

    # Read system prompt
    if os.path.isfile(system_file):
        with open(system_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    else:
        print(f"[WARNING] Missing system prompt: {system_file}")
        system_prompt = ""

    # Read user prompt
    if os.path.isfile(user_file):
        with open(user_file, "r", encoding="utf-8") as f:
            user_prompt = f.read()
    else:
        print(f"[WARNING] Missing user prompt: {user_file}")
        user_prompt = ""

    # Return based on requested format
    if return_dict:
        return {"system": system_prompt, "user": user_prompt}
    else:
        return system_prompt, user_prompt


class LLMClient:
    def __init__(self, model: str = QUERYLLM):
        """
        Initialize the LLM client with a specified model.
        
        Args:
            model: The name of the Ollama model to use (default: "qwen2.5-coder:14b")
        """
        self.debug_enabled = False
        self.model_name = model
        # Basic initialization - we'll set specific parameters when we call
        self.model = Ollama(
            model=model,
            **({"base_url": os.environ["OLLAMA_HOST"]} if "OLLAMA_HOST" in os.environ else {})
        )
        
    def _get_response_text(self, response) -> str:
        """
        Extract text from response, handling different response types.
        
        Args:
            response: Response from the model
            
        Returns:
            Text content as string
        """
        if hasattr(response, 'content'):
            return response.content
        else:
            # If it's already a string, return it directly
            return response
        
    def call(self, 
             prompt: str, 
             system_prompt: Optional[str] = None,
             chat_history: Optional[List[Dict[str, str]]] = None,
             temperature: float = 0.0, 
             max_tokens: int = 512) -> str:
        """
        Call the LLM with proper message formatting.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system instructions
            chat_history: Optional list of previous messages in the format 
                          [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            temperature: Control randomness (0.0 = deterministic, higher = more random)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The model's response as a string
        """
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
            
        # Add chat history if provided
        if chat_history:
            for message in chat_history:
                if message["role"] == "user":
                    messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    messages.append(AIMessage(content=message["content"]))
                elif message["role"] == "system":
                    messages.append(SystemMessage(content=message["content"]))
        
        # Add the current user prompt
        messages.append(HumanMessage(content=prompt))

        # Check if debugging is enabled (you can set this as a class attribute or argument)
        if self.debug_enabled:  # Assuming debug_enabled is a boolean attribute of the class
            print("\n[DEBUG] === Constructed Prompt: ===")
            for msg in messages:  # où 'messages' est ta liste
                role = msg.__class__.__name__.replace('Message', '')  # System / Human
                print(f"\n--- {role} ---\n")
                print(msg.content)
            print(f"\n[DEBUG] === END ===\n")
        
        # Create a new instance with updated parameters for this call
        model = Ollama(
            model=self.model_name,
            temperature=temperature,
            **({"base_url": os.environ["OLLAMA_HOST"]} if "OLLAMA_HOST" in os.environ else {})
        )
        
        try:
            # First try with config parameter
            response = model.invoke(
                messages,
                config={"max_tokens": max_tokens}
            )
        except TypeError:
            try:
                # If that fails, try without config
                response = model.invoke(messages)
            except Exception as e:
                raise RuntimeError(f"Failed to call Ollama: {e}")
                
        return self._get_response_text(response)
    
    def call_with_template(self,
                           system_template: str,
                           user_template: str,
                           temperature: float = 0.0,
                           max_tokens: int = 512,
                           **template_variables) -> str:
        """
        Call the LLM using templates for system and user messages.
        
        Args:
            system_template: Template string for system message with {variable} placeholders
            user_template: Template string for user message with {variable} placeholders
            temperature: Control randomness (0.0 = deterministic, higher = more random)
            max_tokens: Maximum number of tokens to generate
            **template_variables: Variables to insert into the templates
            
        Returns:
            The model's response as a string
        """
        # Create message templates
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(user_template)
        
        # Create the chat prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        
        # Format the messages with variables
        messages = chat_prompt.format_messages(**template_variables)
        
        # Check if debugging is enabled (you can set this as a class attribute or argument)
        if self.debug_enabled:  # Assuming debug_enabled is a boolean attribute of the class
            print("\n[DEBUG] === Constructed Prompt: ===")
            for msg in messages:  # où 'messages' est ta liste
                role = msg.__class__.__name__.replace('Message', '')  # System / Human
                print(f"\n--- {role} ---\n")
                print(msg.content.strip())
            print(f"\n[DEBUG] === END ===\n")

        # Create model instance with temperature
        model = Ollama(
            model=self.model_name,
            temperature=temperature,
            **({"base_url": os.environ["OLLAMA_HOST"]} if "OLLAMA_HOST" in os.environ else {})
               )
        
        try:
            # First try with config parameter
            response = model.invoke(
                messages,
                config={"max_tokens": max_tokens}
            )
        except TypeError:
            try:
                # If that fails, try without config
                response = model.invoke(messages)
            except Exception as e:
                raise RuntimeError(f"Failed to call Ollama: {e}")
                
        return self._get_response_text(response)
    
    def set_model(self, model_name: str) -> None:
        """
        Change the model being used.
        
        Args:
            model_name: The name of the Ollama model to use
        """
        self.model_name = model_name
        self.model = Ollama(
            model=model_name,
            **({"base_url": os.environ["OLLAMA_HOST"]} if "OLLAMA_HOST" in os.environ else {})
            )

    def set_debug(self, dbg: bool) -> None:
        """
        Enable/disable debugging.
        
        Args:
            dbg: The boolean status to set for debug_enabled
        """
        self.debug_enabled = dbg



# Example usage
if __name__ == "__main__":
    # Initialize the client
    llm_client = LLMClient(model=QUERYLLM)
    
    # Example 1: Simple prompt
    response = llm_client.call(
        prompt="Write a function to calculate the Fibonacci sequence",
        system_prompt="You are a helpful coding assistant. Provide clean, well-commented code."
    )
    print(response)
    
    # Example 2: With chat history
    chat_history = [
        {"role": "user", "content": "How do I read a CSV file in Python?"},
        {"role": "assistant", "content": "You can use the csv module or pandas library..."}
    ]
    response = llm_client.call(
        prompt="How would I modify that to handle large files?",
        system_prompt="You are a helpful coding assistant.",
        chat_history=chat_history
    )
    print(response)
    
    # Example 3: Using templates
    response = llm_client.call_with_template(
        system_template="You are an expert {language} programmer. Be concise and efficient.",
        user_template="Write a function to {task}",
        language="Python",
        task="find the prime factors of a number"
    )
    print(response)
