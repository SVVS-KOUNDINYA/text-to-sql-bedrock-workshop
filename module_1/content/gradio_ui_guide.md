# Using the Gradio UI for Text-to-SQL

This guide explains how to use the Gradio UI that has been added to the module 1 notebook for interacting with the diabetes dataset using natural language queries.

## What is Gradio?

[Gradio](https://www.gradio.app/) is a Python library that allows you to quickly create customizable web interfaces for your machine learning models, data analyses, or any Python function. In this case, we're using it to provide an interactive interface for our text-to-SQL functionality.

## Getting Started

1. Run all the notebook cells up to and including the Gradio UI cell.
2. The Gradio interface will appear directly in the notebook.
3. You'll see a text input box where you can type your questions about the diabetes dataset.

## Features

### Conversational Interface

The Gradio UI provides a chat-like interface that:
- Maintains conversation history
- Allows follow-up questions
- Shows both your questions and the system's answers

### Example Questions

The interface includes example questions that you can click on to quickly try out the system:
- How many patients have a BMI over 30?
- What is the average age of patients with diabetes?
- How many patients are over 50 years old and have high blood pressure?
- What is the correlation between BMI and blood glucose levels?
- Show me the top 5 patients with the highest insulin levels

### Query Details

Each answer includes additional information about:
- Execution time
- Number of actions performed
- The SQL query that was generated (if enabled)

### Controls

- **Submit Button**: Send your question to the system
- **Clear Conversation**: Reset the conversation history

## Tips for Effective Questions

1. **Be specific**: Clearly state what information you're looking for
2. **Mention columns**: Reference specific columns from the dataset when relevant
3. **Follow-up questions**: You can ask follow-up questions that reference previous queries
4. **Complex queries**: The system can handle complex questions involving multiple conditions

## Troubleshooting

If you encounter any issues:
1. Make sure all previous notebook cells have been executed
2. Check that the Bedrock model has been properly configured
3. Try clearing the conversation history and starting fresh
4. Ensure your questions are clear and relate to the diabetes dataset

## Technical Details

The Gradio UI uses the same underlying `answer_standalone_question` function that powers the direct notebook queries. It adds a conversational layer that maintains context between questions, allowing for a more natural interaction experience.