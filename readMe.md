# LlamaChat LLM Program

## What is LlamaChat?

LlamaChat is a Python program designed to facilitate conversational interactions with a machine learning model, specifically leveraging the LlamaCpp language model from the LangChain library. It enables users to conduct chat sessions where each interaction is remembered within the session, allowing for context-aware responses. Additionally, the program provides functionalities to save and load chat histories, ensuring a seamless conversational experience across sessions.

## Features

- **Interactive Chat Sessions**: Engage in real-time conversations with the AI, with each input and response remembered for context.
- **Context Management**: Manages the chat context, ensuring the AI's responses are relevant to the ongoing conversation.
- **History Management**: Save chat histories to a file for later retrieval, allowing continuation of previous sessions.
- **History Loading**: Load an existing chat history to resume a past conversation.
- **Customizable Model Configuration**: Configure the underlying LlamaCpp model with custom settings, such as GPU layers, batch size, and temperature.

## Installation

To use LlamaChat History, you need Python installed on your machine, along with the necessary libraries. Follow these steps:

1. Clone the LlamaChat History repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the github cloned directory:
   ```bash
   cd CS-4793-Group-Project
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start a Chat Session**: To begin an interactive chat session, run the script with:
   ```python
   python llama_chat.py
   ```
   Follow the prompts to either start a new chat or continue from an existing history.

2. **Conversing with the AI**: Type your messages into the console. The AI will respond based on the current and past interactions within the session.

3. **Ending a Chat Session**: To end the session, type `exit`. You will be prompted to save the conversation history.

4. **Saving Chat Histories**: If you choose to save the session, it will be stored in the specified history directory.

5. **Resuming a Chat Session**: In future sessions, you can load a previous chat history to continue where you left off.

## Configuration

Modify the `model_path` in the script to point to your LlamaCpp model's location. Adjust model settings such as `n_gpu_layers`, `n_batch`, and `temperature` as needed for optimal performance.

## Dependencies

- Python 3.8+
- LangChain
- UUID
- JSON, os, glob libraries (standard Python libraries)

Ensure you have the latest versions of these dependencies to avoid compatibility issues.

## Contributing

1. Bronson Woods
2. James Jolly
3. Willam Vang

---
