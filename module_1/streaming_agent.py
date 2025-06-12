"""
This module implements a streaming agent for the text-to-SQL system.
It extends the SQLHandler class to capture and stream the agent's thought process to Gradio.
"""

from typing import List, Callable, Any, Dict, Optional
from langchain.callbacks.base import BaseCallbackHandler
import queue
import threading
import time

class StreamingAgentHandler(BaseCallbackHandler):
    """
    Callback handler for streaming agent thoughts to Gradio.
    This handler captures the agent's thought process and streams it to the UI.
    """
    def __init__(self, streaming_callback: Optional[Callable[[str], Any]] = None):
        self._sql_result = []
        self._num_tool_actions = 0
        self._streaming_callback = streaming_callback
        self._thought_process = []
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts running."""
        if self._streaming_callback:
            self._streaming_callback("🤔 Thinking...")
    
    def on_llm_new_token(self, token, **kwargs):
        """Called when LLM produces a new token."""
        if self._streaming_callback and token.strip():
            self._streaming_callback(token)
    
    def on_agent_action(self, action, **kwargs):
        """Runs on agent action."""
        self._num_tool_actions += 1
        if action.tool in ["sql_db_query_checker", "sql_db_query"]:
            self._sql_result.append(action.tool_input)
        
        # Capture and stream the thought process
        thought = f"🔍 Action: {action.tool}\n💭 Input: {action.tool_input}"
        self._thought_process.append(thought)
        if self._streaming_callback:
            self._streaming_callback(thought)
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Called when a chain starts running."""
        if self._streaming_callback and "question" in inputs:
            self._streaming_callback(f"💬 Question: {inputs['question']}")
    
    def on_chain_end(self, outputs, **kwargs):
        """Called when a chain ends running."""
        if self._streaming_callback and "text" in outputs:
            self._streaming_callback(f"📝 Answer: {outputs['text']}")
    
    def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes."""
        if self._streaming_callback:
            self._streaming_callback("✅ Finished reasoning")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when a tool starts running."""
        if self._streaming_callback:
            self._streaming_callback(f"🛠️ Running tool: {serialized['name']}")
    
    def on_tool_end(self, output, **kwargs):
        """Called when a tool ends running."""
        if self._streaming_callback:
            # Truncate long outputs to avoid overwhelming the UI
            display_output = str(output)
            if len(display_output) > 500:
                display_output = display_output[:500] + "... [output truncated]"
            self._streaming_callback(f"📊 Tool result: {display_output}")
    
    def sql_results(self) -> List[str]:
        return self._sql_result

    def num_tool_actions(self) -> int:
        return self._num_tool_actions
        
    def thought_process(self) -> List[str]:
        return self._thought_process


class StreamingChatbot:
    """
    A class to handle streaming updates to a Gradio chatbot.
    This allows for real-time updates of the agent's thought process.
    """
    def __init__(self):
        self.queue = queue.Queue()
        self.thinking = False
        self.lock = threading.Lock()
        self.current_question = ""
        self.current_thoughts = []
        self.final_answer = ""
        
    def start_thinking(self, question: str):
        """Start the thinking process for a new question."""
        with self.lock:
            self.thinking = True
            self.current_question = question
            self.current_thoughts = []
            self.final_answer = ""
            
    def add_thought(self, thought: str):
        """Add a thought to the current thinking process."""
        with self.lock:
            if self.thinking:
                self.current_thoughts.append(thought)
                self.queue.put(thought)
                
    def finish_thinking(self, answer: str):
        """Finish the thinking process with a final answer."""
        with self.lock:
            self.thinking = False
            self.final_answer = answer
            self.queue.put("DONE:" + answer)
            
    def get_updates(self):
        """Get all updates since the last call."""
        updates = []
        while not self.queue.empty():
            updates.append(self.queue.get())
        return updates
    
    def get_current_state(self):
        """Get the current state of the thinking process."""
        with self.lock:
            return {
                "question": self.current_question,
                "thoughts": self.current_thoughts.copy(),
                "answer": self.final_answer,
                "thinking": self.thinking
            }