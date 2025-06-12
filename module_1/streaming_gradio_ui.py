"""
Gradio UI for the text-to-SQL system with streaming agent thoughts.
This module implements a Gradio interface that displays the agent's thought process in real-time.

To use this module:
1. Run this file directly: python streaming_gradio_ui.py
2. Or import the demo object: from streaming_gradio_ui import demo
"""

import os
import sys
import time
import gradio as gr
from typing import List, Tuple, Dict, Any, Optional
import threading
import queue
import itertools

# Add parent directory to path for imports
sys.path.append('../')

# Import necessary libraries
import sqlite3
import pandas as pd
from langchain_aws import ChatBedrock
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import jinja2
import utilities as u

# Custom callback handler for streaming agent thoughts
class StreamingAgentHandler(BaseCallbackHandler):
    """
    Callback handler for streaming agent thoughts to Gradio.
    This handler captures the agent's thought process and streams it to the UI.
    """
    def __init__(self, streaming_callback=None):
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

# Class to manage streaming updates to the Gradio UI
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

# Initialize global variables
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
con = sqlite3.connect("test.db")
jenv = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

is_conversational = True
show_SQL = True

# Initialize LLM and database
llm = ChatBedrock(model_id=model_id, region_name="us-west-2")
db = SQLDatabase.from_uri("sqlite:///test.db")
context = db.get_context()
chain = create_sql_query_chain(llm, db)

# Load data if needed
def setup_db():
    """Set up the database with the diabetes dataset."""
    print("Setting up DB")
    df = pd.read_csv("diabetes.csv")
    df.to_sql(name="patients", con=con, if_exists="replace", index=True)
    con.commit()

def maybe_setup_db():
    """Check if the database needs to be set up and do so if needed."""
    try:
        cur = con.cursor()
        cur.execute("SELECT count(*) FROM patients")
        print(f"Table exists ({cur.fetchone()[0]}), no need to recreate DB")
    except Exception as ex:
        cur = con.cursor()
        if "no such table: patients" in str(ex):
            print(f"Table not there, need to recreate DB")
            setup_db()
        else:
            raise ex

# Make sure the database is set up
maybe_setup_db()

# Extract the CREATE TABLE statement from the database
cur = con.cursor()
cur.execute("SELECT * FROM sqlite_master")
DDL = cur.fetchone()[4]

# Function to decontextualize questions for conversational context
def decontextualize_question(question: str, messages: List[List[str]]) -> str:
    """
    Each message is a list of [question, answer].
    """
    prompt_template = """
    I am going to give you a history of questions and answers, followed by a new question.
    I want you to rewrite to the new question so that it stands alone, not needing the
    historical context to make sense.

    <history>
    {% for x in history %}
      <question>{{ x[0] }}</question>
      <answer>{{ x[1] }}</answer>
    {% endfor %}
    </history>

    Here is the new question:
    <new_question>
    {{question}}
    </new_question>

    You must make the absolute MINIMUM changes required to make the meaning of
    the sentence clear without the context of the history. Make NO other changes.

    Return the rewritten, standalone, question in <r></r> tags.
    """
    prompt = jenv.from_string(prompt_template).render(history=messages, question=question)
    response = llm.invoke(prompt)
    
    # Extract the answer from the response
    answer = response.content
    # Extract content between <r> and </r> tags
    import re
    match = re.search(r'<r>(.*?)</r>', answer, re.DOTALL)
    if match:
        return match.group(1).strip()
    return question  # Return original question if no match

# Create the prompt for the agent
def create_prompt(notes, DDL):
    prompt_template = '''
    Answer the following questions as best you can.

    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    You might find the following tips useful:
    {% for tip in tips %}
      - {{ tip }}
    {% endfor %}

    The database has the following single table:

    {{ table_info }}

    You should NEVER have to use either the sql_db_schema tool or the sql_db_list_tables tool
    as you know the only table is the "patients" table and you know its schema.

    You NEVER can product SELECT statement with no LIMIT clause. You should always have an ORDER BY
    clause and a "LIMIT 20" to avoid returning too many useless results.

    When describing the final result you don't have to describe HOW the SQL statement worked,
    just describe the results.

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}'''
    
    prompt_0 = jenv.from_string(prompt_template).render(tips=notes, table_info=DDL)
    prompt = PromptTemplate.from_template(prompt_0)
    return prompt

# Initialize the streaming chatbot
streaming_chatbot = StreamingChatbot()

# Function to answer a question with streaming updates
def answer_question_streaming(question: str, 
                             messages: List[List[str]], 
                             streaming_callback=None) -> Tuple[str, str]:
    """
    Answer a question with streaming updates to the UI.
    Returns the final answer and additional info.
    """
    start_time = time.time()
    
    if is_conversational and messages:
        question = decontextualize_question(question, messages)
    
    # Create the handler with the streaming callback
    handler = StreamingAgentHandler(streaming_callback)
    
    try:
        # Create the agent with our custom handler
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=db, llm=llm),
            verbose=True,
            prompt=create_prompt([], DDL),
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            callbacks=[handler],
            handle_parsing_errors=True)
        
        # Execute the agent
        for iteration in itertools.count(0):
            try:
                answer = agent_executor.invoke(
                    input={"input": question},
                    config={"callbacks": [handler]}
                )
                
                # Calculate metrics
                duration = time.time() - start_time
                iter_str = f", {iteration} iterations" if iteration > 1 else ""
                history_str = f", history {len(messages):,}" if len(messages) > 0 else ""
                
                # Get SQL result if available
                sql_result = handler.sql_results()[-1].strip() if len(handler.sql_results()) > 0 else None
                SQL_str = f"\n```{sql_result}```" if show_SQL and sql_result else ""
                
                # Return the answer and additional info
                return (
                    answer['output'],
                    f"{duration:.1f} secs, {handler.num_tool_actions():,} actions{iter_str}{history_str} {SQL_str}"
                )
            except ValueError as ex:
                if iteration < 10:
                    print(f"iteration #{iteration}: caught {ex}")
                    print("retrying")
                    if streaming_callback:
                        streaming_callback(f"⚠️ Error: {str(ex)[:100]}... Retrying...")
                else:
                    raise ex
    except Exception as ex:
        print(f"Caught: {ex}")
        if streaming_callback:
            streaming_callback(f"❌ Error: {str(ex)}")
        raise ex

# Store conversation history for the Gradio interface
conversation_history = []

# Function to process a question and update the UI
def process_question(question, chatbot):
    global conversation_history
    
    if not question.strip():
        return "", chatbot
    
    # Start the streaming process
    streaming_chatbot.start_thinking(question)
    
    # Create a list to store thought updates
    thoughts = []
    
    # Define the streaming callback
    def streaming_callback(thought):
        streaming_chatbot.add_thought(thought)
        thoughts.append(thought)
        
        # Update the chatbot with the current thoughts
        current_thoughts = "\n".join(thoughts)
        current_chatbot = chatbot.copy() if chatbot else []
        
        # Add or update the current question's thoughts
        if current_chatbot and current_chatbot[-1][0] == question:
            current_chatbot[-1][1] = f"🧠 **Thinking...**\n\n{current_thoughts}"
        else:
            current_chatbot.append([question, f"🧠 **Thinking...**\n\n{current_thoughts}"])
        
        # Yield updates to the UI
        return "", current_chatbot
    
    # Start a thread to process the question
    def process_thread():
        try:
            # Process the question with streaming updates
            answer, extra_info = answer_question_streaming(
                question, 
                conversation_history,
                streaming_callback
            )
            
            # Update conversation history
            conversation_history.append([question, answer])
            
            # Format the response with the answer and additional info
            response = f"{answer}\n\n---\n*Query details: {extra_info}*"
            
            # Signal that thinking is complete
            streaming_chatbot.finish_thinking(response)
        except Exception as e:
            # Handle any errors
            error_msg = f"❌ Error: {str(e)}"
            streaming_chatbot.finish_thinking(error_msg)
    
    # Start the processing thread
    thread = threading.Thread(target=process_thread)
    thread.start()
    
    # Return a function that will yield updates
    def get_updates():
        while True:
            # Check for updates
            updates = streaming_chatbot.get_updates()
            if updates:
                for update in updates:
                    if update.startswith("DONE:"):
                        # Final answer received
                        final_answer = update[5:]
                        current_chatbot = chatbot.copy() if chatbot else []
                        
                        # Update the chatbot with the final answer
                        if current_chatbot and current_chatbot[-1][0] == question:
                            current_chatbot[-1][1] = final_answer
                        else:
                            current_chatbot.append([question, final_answer])
                        
                        # Return the final state
                        return "", current_chatbot
                
                # Return intermediate updates
                current_thoughts = "\n".join(thoughts)
                current_chatbot = chatbot.copy() if chatbot else []
                
                if current_chatbot and current_chatbot[-1][0] == question:
                    current_chatbot[-1][1] = f"🧠 **Thinking...**\n\n{current_thoughts}"
                else:
                    current_chatbot.append([question, f"🧠 **Thinking...**\n\n{current_thoughts}"])
                
                return "", current_chatbot
            
            # Sleep briefly to avoid consuming too much CPU
            time.sleep(0.1)
    
    # Return the update function
    return get_updates

def clear_history():
    global conversation_history
    conversation_history = []
    return "", []

# Create the Gradio interface
with gr.Blocks(title="Text-to-SQL on Diabetes Dataset with Streaming Thoughts") as demo:
    gr.Markdown("# Text-to-SQL on Diabetes Dataset with Streaming Agent Thoughts")
    gr.Markdown("Ask questions about the diabetes dataset in natural language. The system will convert your question to SQL and return the answer. You'll see the agent's thought process in real-time!")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Conversation", height=600)
            with gr.Row():
                question_input = gr.Textbox(label="Your question", placeholder="e.g., How many patients have a BMI over 30?", lines=2)
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear Conversation")
    
    # Set up event handlers
    submit_btn.click(process_question, inputs=[question_input, chatbot], outputs=[question_input, chatbot])
    question_input.submit(process_question, inputs=[question_input, chatbot], outputs=[question_input, chatbot])
    clear_btn.click(clear_history, inputs=[], outputs=[question_input, chatbot])
    
    gr.Markdown("### Example Questions:")
    examples = gr.Examples(
        examples=[
            "How many patients have a BMI over 30?",
            "What is the average age of patients with diabetes?",
            "How many patients are over 50 years old and have high blood pressure?",
            "What is the correlation between BMI and blood glucose levels?",
            "Show me the top 5 patients with the highest insulin levels"
        ],
        inputs=question_input
    )

# Launch the interface if this file is run directly
if __name__ == "__main__":
    demo.queue().launch(share=False)
