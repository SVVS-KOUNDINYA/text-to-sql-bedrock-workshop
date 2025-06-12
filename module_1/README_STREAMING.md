# Streaming Agent Thoughts in Text-to-SQL

This extension to the module_1 text-to-SQL system adds real-time streaming of the agent's thought process to the Gradio UI. This allows users to see how the agent reasons through questions, formulates SQL queries, and arrives at answers.

## Features

- **Real-time Thought Process**: See the agent's reasoning as it happens
- **SQL Query Visibility**: Watch as SQL queries are formulated and executed
- **Interactive UI**: Ask questions and see the thought process unfold
- **Conversational Context**: Follow-up questions maintain context from previous interactions

## How to Use

### Option 1: Run the Streaming UI Directly

```bash
cd module_1
python streaming_gradio_ui.py
```

This will launch a Gradio web interface where you can:
1. Enter questions about the diabetes dataset
2. Watch the agent's thought process in real-time
3. See the final answer and SQL query

### Option 2: Import in Your Own Code

```python
from module_1.streaming_gradio_ui import demo

# Launch the demo
demo.launch()
```

## Implementation Details

The streaming functionality is implemented using:

1. **StreamingAgentHandler**: A custom callback handler that captures the agent's thoughts
   - Extends LangChain's `BaseCallbackHandler`
   - Captures events like `on_llm_start`, `on_agent_action`, etc.
   - Streams updates to the UI in real-time

2. **StreamingChatbot**: A class to manage streaming updates to the Gradio UI
   - Maintains a queue of thought updates
   - Handles the state of the thinking process
   - Provides methods to start thinking, add thoughts, and finish thinking

## Example Questions

Try asking questions like:
- "How many patients have a BMI over 30?"
- "What is the average age of patients with diabetes?"
- "How many patients are over 50 years old and have high blood pressure?"
- "What is the correlation between BMI and blood glucose levels?"
- "Show me the top 5 patients with the highest insulin levels"

## Testing

Run the tests to verify the streaming functionality:

```bash
cd module_1
python -m unittest test_streaming_gradio_ui.py
```