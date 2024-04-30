import tkinter as tk
import tkinter.scrolledtext as scrolledtext
from tkinter import filedialog, messagebox
import threading
import json
import os
import uuid
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class LlamaChatHistory:
    def __init__(self, master, model_path):
        self.master = master
        self.model_path = model_path
        self.history = []
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llama_model = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=-1,
            n_batch=512,
            f16_kv=True,
            temperature=0.5,
            max_tokens=2000,
            n_ctx=4096,
            top_p=1,
            callback_manager=self.callback_manager,
            verbose=False
        )
        self.current_history_file = None
        self.create_widgets()

    def create_widgets(self):
        self.master.title("Llama Chat GUI")
        self.text_area = scrolledtext.ScrolledText(self.master, state='disabled')
        self.text_area.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        self.input_user = tk.Entry(self.master, width=50)
        self.input_user.grid(row=1, column=0, padx=10, pady=10)

        self.send_button = tk.Button(self.master, text="Send", command=self.send_input)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.save_button = tk.Button(self.master, text="Save Chat", command=self.save_chat)
        self.save_button.grid(row=1, column=2, padx=10, pady=10)

        self.load_button = tk.Button(self.master, text="Load Chat", command=self.load_chat)
        self.load_button.grid(row=1, column=3, padx=10, pady=10)

    def send_input(self):
        user_input = self.input_user.get()
        if user_input:
            self.input_user.delete(0, tk.END)
            self.display_message("You: " + user_input)
            threading.Thread(target=self.handle_response, args=(user_input,)).start()

    def handle_response(self, user_input):
        response = self.generate_response(user_input)
        self.display_message("Bot: " + response)

    def display_message(self, message):
        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, message + '\n')
        self.text_area.config(state='disabled')
        self.text_area.yview(tk.END)

    def generate_response(self, user_input):
        prompt_template = (
            "[INST] <<SYS>>"
            "You are a helpful, respectful, and honest assistant..."
            "<</SYS>>"
            "{user_input}[/INST]"
        )
        full_prompt = prompt_template + "\n".join(["Q: " + p + "\nA: " + r for p, r in self.history])
        full_prompt += "\nQ: " + user_input + "[/INST]"
        response = self.llama_model(full_prompt)
        self.add_to_history(user_input, response)
        return response

    def add_to_history(self, prompt, response):
        self.history.append((prompt, response))

    def save_chat(self):
        directory = filedialog.askdirectory()
        if directory:
            self.save_history(directory)

    def save_history(self, directory):
        if not self.current_history_file:
            chat_id = str(uuid.uuid4())
            self.current_history_file = os.path.join(directory, f"chat_history_{chat_id}.json")
        with open(self.current_history_file, 'w') as file:
            json.dump(self.history, file)

    def load_chat(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            self.load_history(file_path)

    def load_history(self, file_path):
        try:
            with open(file_path, 'r') as file:
                self.history = json.load(file)
            self.current_history_file = file_path  # Store the current file path
            self.text_area.config(state='normal')
            self.text_area.delete(1.0, tk.END)
            for prompt, response in self.history:
                self.display_message("Q: " + prompt)
                self.display_message("A: " + response)
            self.text_area.config(state='disabled')
        except (FileNotFoundError, json.JSONDecodeError) as e:
            messagebox.showerror("Error", "Failed to load chat history")

# Usage
root = tk.Tk()
model_path = "/Users/bronsonwoods/AA_Projects/CS-4793-Group-Project/Models/llama-2-7b-chat.Q4_K_M.gguf"
app = LlamaChatHistory(root, model_path)
root.mainloop()
