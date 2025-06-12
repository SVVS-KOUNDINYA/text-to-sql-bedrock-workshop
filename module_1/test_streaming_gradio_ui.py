"""
Test script for the Streaming Gradio UI in module 1.
This script tests the functionality of the streaming agent thoughts in the Gradio UI.
"""

import unittest
from unittest.mock import patch, MagicMock, call

# Import the streaming agent classes for testing
from streaming_gradio_ui import StreamingAgentHandler, StreamingChatbot

class TestStreamingAgentHandler(unittest.TestCase):
    """Test the StreamingAgentHandler class."""
    
    def test_initialization(self):
        """Test that the handler initializes correctly."""
        handler = StreamingAgentHandler()
        self.assertEqual(handler._sql_result, [])
        self.assertEqual(handler._num_tool_actions, 0)
        self.assertEqual(handler._thought_process, [])
        self.assertIsNone(handler._streaming_callback)
        
        # Test with a callback
        mock_callback = MagicMock()
        handler = StreamingAgentHandler(streaming_callback=mock_callback)
        self.assertEqual(handler._streaming_callback, mock_callback)
    
    def test_on_llm_start(self):
        """Test the on_llm_start method."""
        mock_callback = MagicMock()
        handler = StreamingAgentHandler(streaming_callback=mock_callback)
        
        # Call the method
        handler.on_llm_start({}, ["test prompt"])
        
        # Verify the callback was called
        mock_callback.assert_called_once_with("🤔 Thinking...")
    
    def test_on_agent_action(self):
        """Test the on_agent_action method."""
        mock_callback = MagicMock()
        handler = StreamingAgentHandler(streaming_callback=mock_callback)
        
        # Create a mock action
        mock_action = MagicMock()
        mock_action.tool = "sql_db_query"
        mock_action.tool_input = "SELECT * FROM patients LIMIT 10"
        
        # Call the method
        handler.on_agent_action(mock_action)
        
        # Verify the results
        self.assertEqual(handler._num_tool_actions, 1)
        self.assertEqual(handler._sql_result, ["SELECT * FROM patients LIMIT 10"])
        self.assertEqual(len(handler._thought_process), 1)
        self.assertIn("sql_db_query", handler._thought_process[0])
        
        # Verify the callback was called
        mock_callback.assert_called_once()
        self.assertIn("Action: sql_db_query", mock_callback.call_args[0][0])
    
    def test_sql_results_and_num_tool_actions(self):
        """Test the sql_results and num_tool_actions methods."""
        handler = StreamingAgentHandler()
        
        # Add some test data
        handler._sql_result = ["SELECT * FROM patients"]
        handler._num_tool_actions = 3
        
        # Verify the methods return the correct values
        self.assertEqual(handler.sql_results(), ["SELECT * FROM patients"])
        self.assertEqual(handler.num_tool_actions(), 3)
    
    def test_thought_process(self):
        """Test the thought_process method."""
        handler = StreamingAgentHandler()
        
        # Add some test data
        handler._thought_process = ["Thought 1", "Thought 2"]
        
        # Verify the method returns the correct value
        self.assertEqual(handler.thought_process(), ["Thought 1", "Thought 2"])


class TestStreamingChatbot(unittest.TestCase):
    """Test the StreamingChatbot class."""
    
    def test_initialization(self):
        """Test that the chatbot initializes correctly."""
        chatbot = StreamingChatbot()
        self.assertFalse(chatbot.thinking)
        self.assertEqual(chatbot.current_question, "")
        self.assertEqual(chatbot.current_thoughts, [])
        self.assertEqual(chatbot.final_answer, "")
    
    def test_start_thinking(self):
        """Test the start_thinking method."""
        chatbot = StreamingChatbot()
        
        # Call the method
        chatbot.start_thinking("Test question")
        
        # Verify the state
        self.assertTrue(chatbot.thinking)
        self.assertEqual(chatbot.current_question, "Test question")
        self.assertEqual(chatbot.current_thoughts, [])
        self.assertEqual(chatbot.final_answer, "")
    
    def test_add_thought(self):
        """Test the add_thought method."""
        chatbot = StreamingChatbot()
        
        # Start thinking
        chatbot.start_thinking("Test question")
        
        # Add a thought
        chatbot.add_thought("Test thought")
        
        # Verify the state
        self.assertEqual(chatbot.current_thoughts, ["Test thought"])
        self.assertFalse(chatbot.queue.empty())
        self.assertEqual(chatbot.queue.get(), "Test thought")
        
        # Test adding a thought when not thinking
        chatbot.thinking = False
        chatbot.add_thought("Another thought")
        
        # Verify the thought wasn't added
        self.assertEqual(chatbot.current_thoughts, ["Test thought"])
        self.assertTrue(chatbot.queue.empty())
    
    def test_finish_thinking(self):
        """Test the finish_thinking method."""
        chatbot = StreamingChatbot()
        
        # Start thinking
        chatbot.start_thinking("Test question")
        
        # Finish thinking
        chatbot.finish_thinking("Test answer")
        
        # Verify the state
        self.assertFalse(chatbot.thinking)
        self.assertEqual(chatbot.final_answer, "Test answer")
        self.assertFalse(chatbot.queue.empty())
        self.assertEqual(chatbot.queue.get(), "DONE:Test answer")
    
    def test_get_updates(self):
        """Test the get_updates method."""
        chatbot = StreamingChatbot()
        
        # Add some items to the queue
        chatbot.queue.put("Update 1")
        chatbot.queue.put("Update 2")
        
        # Get the updates
        updates = chatbot.get_updates()
        
        # Verify the updates
        self.assertEqual(updates, ["Update 1", "Update 2"])
        self.assertTrue(chatbot.queue.empty())
    
    def test_get_current_state(self):
        """Test the get_current_state method."""
        chatbot = StreamingChatbot()
        
        # Set up some state
        chatbot.thinking = True
        chatbot.current_question = "Test question"
        chatbot.current_thoughts = ["Thought 1", "Thought 2"]
        chatbot.final_answer = "Test answer"
        
        # Get the state
        state = chatbot.get_current_state()
        
        # Verify the state
        self.assertEqual(state, {
            "question": "Test question",
            "thoughts": ["Thought 1", "Thought 2"],
            "answer": "Test answer",
            "thinking": True
        })


@patch('gradio.Blocks')
@patch('gradio.Chatbot')
@patch('gradio.Textbox')
@patch('gradio.Button')
class TestStreamingGradioUI(unittest.TestCase):
    """Test the Streaming Gradio UI."""
    
    @patch('streaming_gradio_ui.StreamingChatbot')
    @patch('streaming_gradio_ui.answer_question_streaming')
    @patch('threading.Thread')
    def test_process_question(self, mock_thread, mock_answer, mock_chatbot, mock_button, mock_textbox, mock_chatbot_ui, mock_blocks):
        """Test the process_question function."""
        # Import the necessary code (with mocked dependencies)
        with patch('gradio.Markdown'), patch('gradio.Row'), patch('gradio.Column'), patch('gradio.Examples'):
            # Create a namespace to hold our variables
            from streaming_gradio_ui import process_question
            
            # Mock the streaming chatbot
            mock_chatbot_instance = MagicMock()
            mock_chatbot.return_value = mock_chatbot_instance
            
            # Mock the answer_question_streaming function
            mock_answer.return_value = ("Test answer", "1.0 secs, 3 actions")
            
            # Test with an empty question
            result = process_question("", [])
            self.assertEqual(result, ("", []))
            
            # Test with a valid question
            chatbot = []
            result = process_question("Test question", chatbot)
            
            # Verify the streaming chatbot was used
            mock_chatbot_instance.start_thinking.assert_called_with("Test question")
            
            # Verify a thread was started
            mock_thread.assert_called()
            
            # The result should be a function
            self.assertTrue(callable(result))


if __name__ == '__main__':
    unittest.main()