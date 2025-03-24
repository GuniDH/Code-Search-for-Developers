import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import os
import threading
from semantic_code_search import SemanticCodeSearch
import json

# Guni Deyo Haness 
# 2025-03-20    
# This is the UI for the semantic code search application.
# It allows you to build an index of code snippets from a source directory
# and search for code snippets by natural language queries.

class SemanticCodeSearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Semantic Code Search")
        self.root.geometry("1000x700")
        
        self.search_engine = None
        self.setup_ui()
    
    def setup_ui(self):
        # Frame for configuration
        config_frame = tk.Frame(self.root, pady=10)
        config_frame.pack(fill=tk.X, padx=10)
        
        # Source directory selection
        tk.Label(config_frame, text="Source Directory:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.source_dir_var = tk.StringVar()
        tk.Entry(config_frame, textvariable=self.source_dir_var, width=50).grid(row=0, column=1, padx=5)
        tk.Button(config_frame, text="Browse", command=self.browse_source_dir).grid(row=0, column=2, padx=5)
        
        # API Token input
        tk.Label(config_frame, text="OpenAI API Token:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.token_var = tk.StringVar()
        tk.Entry(config_frame, textvariable=self.token_var, width=50, show="*").grid(row=1, column=1, padx=5)
        
        # Build Index button
        tk.Button(config_frame, text="Build Index", command=self.build_index).grid(row=1, column=2, padx=5)
        
        # Progress bar - ALWAYS VISIBLE
        progress_frame = tk.Frame(self.root, pady=10, padx=10, relief=tk.GROOVE, bd=1)
        progress_frame.pack(fill=tk.X, padx=10)
        
        tk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        status_frame = tk.Frame(progress_frame)
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress_percent = tk.StringVar(value="0%")
        tk.Label(status_frame, textvariable=self.progress_percent, width=5).pack(side=tk.LEFT, padx=5)
        
        self.progress_status = tk.StringVar(value="Ready")
        tk.Label(status_frame, textvariable=self.progress_status, width=30, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        
        # Cancel button for index building
        self.cancel_button = tk.Button(progress_frame, text="Cancel", command=self.cancel_build)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        self.cancel_button.config(state=tk.DISABLED)
        
        # Search interface
        search_frame = tk.Frame(self.root, pady=10)
        search_frame.pack(fill=tk.X, padx=10)
        
        tk.Label(search_frame, text="Search Query:").pack(side=tk.LEFT, padx=5)
        self.query_var = tk.StringVar()
        tk.Entry(search_frame, textvariable=self.query_var, width=50).pack(side=tk.LEFT, padx=5)
        tk.Button(search_frame, text="Search", command=self.search).pack(side=tk.LEFT, padx=5)
        
        # Results count
        tk.Label(search_frame, text="Results:").pack(side=tk.LEFT, padx=5)
        self.results_count_var = tk.IntVar(value=5)
        tk.Spinbox(search_frame, from_=1, to=20, width=3, textvariable=self.results_count_var).pack(side=tk.LEFT)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Results area
        results_frame = tk.Frame(self.root)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Split view with resizable pane
        self.paned_window = tk.PanedWindow(results_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Results list on the left
        list_frame = tk.Frame(self.paned_window, width=300)
        self.paned_window.add(list_frame, minsize=200)
        
        tk.Label(list_frame, text="Search Results:").pack(anchor=tk.W)
        
        # Add a scrollbar to the listbox
        list_scroll_frame = tk.Frame(list_frame)
        list_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_listbox = tk.Listbox(list_scroll_frame, yscrollcommand=scrollbar.set)
        self.results_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_listbox.yview)
        
        self.results_listbox.bind('<<ListboxSelect>>', self.on_result_select)
        
        # Code display on the right
        code_frame = tk.Frame(self.paned_window)
        self.paned_window.add(code_frame, minsize=300)
        
        tk.Label(code_frame, text="Code:").pack(anchor=tk.W)
        self.code_display = scrolledtext.ScrolledText(code_frame, wrap=tk.WORD, font=("Courier New", 10))
        self.code_display.pack(fill=tk.BOTH, expand=True)
        
        # Store search results
        self.search_results = []
        self.build_thread = None
        self.cancel_requested = False
    
    def browse_source_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.source_dir_var.set(directory)
    
    def update_progress(self, value):
        """Update progress bar"""
        if not self.cancel_requested:
            self.progress["value"] = value
            self.progress_percent.set(f"{value}%")
            
            # Update status text based on progress
            if value < 5:
                self.progress_status.set("Finding source files...")
            elif value < 50:
                self.progress_status.set("Extracting code snippets...")
            else:
                self.progress_status.set("Generating embeddings...")
                
            # Force update the UI
            self.root.update_idletasks()
    
    def cancel_build(self):
        """Request cancellation of the build process"""
        self.cancel_requested = True
        self.progress_status.set("Cancelling...")
        self.status_var.set("Cancelling build process...")
        # The build thread checks the cancel_requested flag periodically
    
    def build_index(self):
        source_dir = self.source_dir_var.get()
        token = self.token_var.get()
        
        if not source_dir or not os.path.isdir(source_dir):
            messagebox.showerror("Error", "Please select a valid source directory")
            return
        
        if not token:
            messagebox.showerror("Error", "Please enter your OpenAI API token")
            return
        
        # Reset cancel flag
        self.cancel_requested = False
        
        # Reset progress bar
        self.progress["value"] = 0
        self.progress_percent.set("0%")
        self.progress_status.set("Starting...")
        
        # Enable cancel button
        self.cancel_button.config(state=tk.NORMAL)
        
        self.status_var.set("Building index... This may take a while")
        self.root.update_idletasks()
        
        # Run in a separate thread to not freeze the UI
        def build_task():
            try:
                # Initialize search engine
                self.search_engine = SemanticCodeSearch(source_dir, token)
                
                # Define a progress callback function
                def progress_callback(value):
                    # Check if cancellation is requested
                    if self.cancel_requested:
                        raise InterruptedError("Build process cancelled by user")
                    self.root.after(0, lambda: self.update_progress(value))
                
                # Build the index with progress updates
                self.search_engine.build_index(
                    force_rebuild=True, 
                    progress_callback=progress_callback
                )
                
                # Update UI on completion
                if not self.cancel_requested:
                    self.root.after(0, lambda: self.status_var.set(
                        f"Index built with {len(self.search_engine.code_snippets)} snippets"
                    ))
                    self.root.after(0, lambda: self.progress_status.set("Complete!"))
                
            except InterruptedError as e:
                self.root.after(0, lambda: self.status_var.set(str(e)))
                self.root.after(0, lambda: self.progress_status.set("Cancelled"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.status_var.set("Error building index"))
                self.root.after(0, lambda: self.progress_status.set("Error"))
            finally:
                # Disable cancel button when done
                self.root.after(0, lambda: self.cancel_button.config(state=tk.DISABLED))
        
        self.build_thread = threading.Thread(target=build_task)
        self.build_thread.daemon = True
        self.build_thread.start()
    
    def search(self):
        if not self.search_engine:
            messagebox.showerror("Error", "Please build the index first")
            return
        
        query = self.query_var.get()
        if not query:
            messagebox.showerror("Error", "Please enter a search query")
            return
        
        top_k = self.results_count_var.get()
        self.status_var.set("Searching...")
        self.root.update_idletasks()
        
        # Run search in a separate thread
        def search_task():
            try:
                results = self.search_engine.search(query, top_k)
                self.root.after(0, lambda: self.update_results(results))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.status_var.set("Error during search"))
        
        threading.Thread(target=search_task, daemon=True).start()
    
    def update_results(self, results):
        self.search_results = results
        self.results_listbox.delete(0, tk.END)
        
        for i, result in enumerate(results):
            filename = os.path.basename(result['file'])
            self.results_listbox.insert(tk.END, f"{i+1}. {filename}: {result['name']}")
        
        self.status_var.set(f"Found {len(results)} results")
        
        # Select the first result
        if results:
            self.results_listbox.selection_set(0)
            self.on_result_select(None)
    
    def on_result_select(self, event):
        selection = self.results_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index < len(self.search_results):
            result = self.search_results[index]
            
            # Update code display
            self.code_display.delete(1.0, tk.END)
            header = f"File: {result['file']}\nFunction: {result['name']}\nSimilarity: {result['similarity']:.4f}\n"
            self.code_display.insert(tk.END, header + "\n" + "-" * 80 + "\n\n")
            self.code_display.insert(tk.END, result['code'])
            
            # Highlight syntax (basic)
            self.highlight_syntax()
    
    def load_keywords(self,keywords_file='language_keywords.json'):
        """
        Load language keywords from a JSON file
        
        :param keywords_file: Path to JSON file containing language keywords
        :return: Dictionary of language keywords
        """
        try:
            with open(keywords_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Keywords file {keywords_file} not found. Using keywords only for c and cpp.")
            return {
                "c_cpp": {
                    "keywords": [
                        "int", "char", "float", "double", "void", "unsigned", "signed",
                        "const", "static", "struct", "union", "enum", "typedef",
                        "if", "else", "for", "while", "do", "switch", "case", "default",
                        "break", "continue", "return", "goto", "sizeof"
                    ]
                }
            }

    def configure_syntax_tags(self,code_text):
        """
        Configure syntax highlighting tags for the text widget
        
        :param code_text: Tkinter Text widget
        """
        # Define tags for syntax highlighting
        code_text.tag_configure("keyword", foreground="blue")
        code_text.tag_configure("string", foreground="green")
        code_text.tag_configure("comment", foreground="gray")
        code_text.tag_configure("function", foreground="purple")

    def highlight_syntax(self, keywords_file='language_keywords.json'):
        """
        Apply syntax highlighting for all available keywords
        
        :param keywords_file: Path to JSON file containing language keywords
        """
        code_text = self.code_display
        
        # Load keywords
        language_keywords = self.load_keywords(keywords_file)
        
        # Configure syntax highlighting tags
        self.configure_syntax_tags(code_text)
        
        # Collect all unique keywords from all languages
        all_keywords = set()
        for language_data in language_keywords.values():
            all_keywords.update(language_data.get('keywords', []))
        
        # Apply syntax highlighting
        for keyword in all_keywords:
            start_pos = "1.0"
            while True:
                start_pos = code_text.search(r'\y' + keyword + r'\y', start_pos, tk.END, regexp=True)
                if not start_pos:
                    break
                end_pos = f"{start_pos}+{len(keyword)}c"
                code_text.tag_add("keyword", start_pos, end_pos)
                start_pos = end_pos
                
def main():
    root = tk.Tk()
    app = SemanticCodeSearchUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()