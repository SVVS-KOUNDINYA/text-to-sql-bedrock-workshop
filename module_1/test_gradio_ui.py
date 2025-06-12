"""
Test script for the Gradio UI in module 1.
This script tests the basic functionality of the Gradio UI components.
"""

import unittest
from unittest.mock import patch, MagicMock

# Mock the required imports and functions
@patch('gradio.Blocks')
@patch('gradio.Chatbot')
@patch('gradio.Textbox')
@patch('gradio.Button')
class TestGradioUI(unittest.TestCase):
    
    def test_process_question_function(self, mock_button, mock_textbox, mock_chatbot, mock_blocks):
        """Test that the process_question function works correctly."""
        # Import the necessary code (with mocked dependencies)
        with patch('gradio.Markdown'), patch('gradio.Row'), patch('gradio.Column'), patch('gradio.Examples'):
            # Create a namespace to hold our variables
            namespace = {}
            
            # Define the mocked answer_standalone_question function
            def mock_answer_standalone_question(question, history):
                return f"Answer to: {question}", "1.0 secs, 3 actions"
            
            # Execute the code with our mocked function
            exec("""
import gradio as gr

# Store conversation history for the Gradio interface
conversation_history = []

def process_question(question):
    global conversation_history
    
    # Use the existing function to answer the question
    answer, extra_info = answer_standalone_question(question, conversation_history)
    
    # Update conversation history
    conversation_history.append([question, answer])
    
    # Format the response with the answer and additional info
    response = f"{answer}\\n\\n---\\n*Query details: {extra_info}*"
    
    # Return the formatted response and update the chat history for display
    chat_history = [[q, a] for q, a in conversation_history]
    return "", chat_history

def clear_history():
    global conversation_history
    conversation_history = []
    return "", []
            """, {"answer_standalone_question": mock_answer_standalone_question, "gr": MagicMock(), **namespace})
            
            # Extract the functions from the namespace
            process_question = namespace.get('process_question')
            clear_history = namespace.get('clear_history')
            
            # Test process_question function
            question = "How many patients have diabetes?"
            _, chat_history = process_question(question)
            
            # Verify the function worked correctly
            self.assertEqual(len(chat_history), 1)
            self.assertEqual(chat_history[0][0], question)
            self.assertEqual(chat_history[0][1], f"Answer to: {question}")
            
            # Test clear_history function
            _, chat_history = clear_history()
            self.assertEqual(len(chat_history), 0)
    
    def test_gradio_components_creation(self, mock_button, mock_textbox, mock_chatbot, mock_blocks):
        """Test that all Gradio components are created."""
        # Mock the gr.Blocks context manager
        blocks_instance = MagicMock()
        mock_blocks.return_value.__enter__.return_value = blocks_instance
        
        # Execute the code that creates the Gradio interface
        with patch('gradio.Markdown') as mock_markdown, \
             patch('gradio.Row') as mock_row, \
             patch('gradio.Column') as mock_column, \
             patch('gradio.Examples') as mock_examples:
            
            # Create a namespace to hold our variables
            namespace = {}
            
            # Execute the code with mocked components
            exec("""
import gradio as gr

# Create the Gradio interface
with gr.Blocks(title="Text-to-SQL on Diabetes Dataset") as demo:
    gr.Markdown("# Text-to-SQL on Diabetes Dataset")
    gr.Markdown("Ask questions about the diabetes dataset in natural language. The system will convert your question to SQL and return the answer.")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Conversation", height=400)
            with gr.Row():
                question_input = gr.Textbox(label="Your question", placeholder="e.g., How many patients have a BMI over 30?", lines=2)
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear Conversation")
    
    # Set up event handlers
    submit_btn.click(process_question, inputs=[question_input], outputs=[question_input, chatbot])
    question_input.submit(process_question, inputs=[question_input], outputs=[question_input, chatbot])
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
            """, {"gr": MagicMock(), "process_question": lambda x: (None, None), "clear_history": lambda: (None, None), **namespace})
            
            # Verify that all components were created
            self.assertTrue(mock_markdown.called)
            self.assertTrue(mock_row.called)
            self.assertTrue(mock_column.called)
            self.assertTrue(mock_chatbot.called)
            self.assertTrue(mock_textbox.called)
            self.assertTrue(mock_button.called)
            self.assertTrue(mock_examples.called)

if __name__ == '__main__':
    unittest.main()