import json
import glob
import os
import uuid
from datetime import datetime
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class LlamaChat_history:
    def __init__(self, model_path):
        self.model_path = model_path
        self.history = []
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llama_model = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=0,
            n_batch=512, 
            f16_kv=True,
            temperature=0.5,
            max_tokens=2000,
            n_ctx=4096,
            top_p=1,
            callback_manager=self.callback_manager,
            verbose=False,
        )
        self.current_history_file = None


    def add_to_history(self, prompt, response):
        self.history.append((prompt, response))
        
    def start_new_chat(self):
        self.history = []
        self.current_history_file = None


    def generate_response(self, user_input):
        prompt_template = (
            "[INST] <<SYS>>"
            "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            "<</SYS>>"
            "{user_input}[/INST]"
        )

        # Combine the instruction part with the conversation history and user input
        full_prompt = prompt_template + "\n".join(["Q: " + p + "\nA: " + r for p, r in self.history])
        full_prompt += "\nQ: " + user_input + "[/INST]"

        # Generate the response using the model
        response = self.llama_model(full_prompt)

        # Add the interaction to history
        self.add_to_history(user_input, response)
        return response

    def save_history(self, directory):
        if not self.current_history_file:
            chat_id = str(uuid.uuid4())
            self.current_history_file = os.path.join(directory, f"chat_history_{chat_id}.json")
        with open(self.current_history_file, 'w') as file:
            json.dump(self.history, file)

    def load_history(self, file_path):
        try:
            with open(file_path, 'r') as file:
                self.history = json.load(file)
            self.current_history_file = file_path 
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = []
            self.current_history_file = None

    
    def count_tokens(self, text):
        return len(text.split())
    
    def list_chat_histories(self, directory):
        """Lists all chat history files in the given directory."""
        return glob.glob(os.path.join(directory, "chat_history_*.json"))

    def interactive_chat(self, history_dir="/ChatHistory"):
        print("Llama Chat Initialized. Type 'exit' to end the conversation.")

        # List available chat histories
        available_histories = self.list_chat_histories(history_dir)
        if available_histories:
            print("Available chat histories:")
            for i, filename in enumerate(available_histories, 1):
                print(f"{i}. {filename}")

            choice = input("Select a chat history number to load or press enter to start a new chat: ")
            if choice.isdigit() and 1 <= int(choice) <= len(available_histories):
                selected_history = available_histories[int(choice) - 1]
                self.load_history(selected_history)
            else:
                print("Starting a new chat session.")

        while True:
            user_input = input("You: ")
            if user_input == "exit":
                break
            
            response = self.generate_response(user_input)

        # Save the conversation with an auto-generated chat ID
        save_action = input("Save this conversation? (yes/no): ")
        if save_action.lower() == 'yes':
            self.save_history(history_dir)
# Usage
model_path = "/Models/llama-2-7b-chat.Q4_K_M.gguf"
llama_chat = LlamaChat_history(model_path=model_path)
llama_chat.interactive_chat()
