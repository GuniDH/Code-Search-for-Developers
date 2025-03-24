import os
import glob
import re
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import argparse
import tiktoken

# Author: Guni 
# 2025-03-20    
# This is the semantic code search engine.
# It allows you to build an index of code snippets from a source directory
# and search for code snippets by natural language queries. 

class SemanticCodeSearch:
    def __init__(self, source_dir, token, model="text-embedding-3-small"):
        """
        Initialize the semantic code search engine.
        
        Args:
            source_dir (str): Directory containing the source code files
            token (str): OpenAI API token
            model (str): OpenAI embedding model
        """
        self.source_dir = source_dir
        self.model = model
        self.client = OpenAI(api_key=token)
        self.code_snippets = []
        self.embeddings = []
        self.db_file = "code_embeddings.json"
        self.max_tokens = 8000  # Set a conservative limit for embedding context
        self.encoding = tiktoken.get_encoding("cl100k_base")  # For token counting
        
    def count_tokens(self, text):
        """Count tokens in a text string."""
        return len(self.encoding.encode(text))
    
    def truncate_to_token_limit(self, text, max_tokens=None):
        """Truncate text to fit within token limit."""
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        return self.encoding.decode(tokens[:max_tokens])
    
    def extract_snippets(self, file_path):
        """
        Extract code snippets from a file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            list: List of dictionaries containing file path, function name and code
        """
        snippets = []
        
        # Read the file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return snippets
        
        # For C/C++ files, extract functions and structs
        if file_path.endswith(('.c', '.h', '.cpp', '.hpp')):
            # Simple regex to capture C functions - not perfect but works for demonstration
            function_pattern = r'(\w+\s+\w+\s*\([^)]*\)\s*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|;))'
            functions = re.findall(function_pattern, content)
            
            for i, func in enumerate(functions):
                # Cleanup the function
                func = func.strip()
                if not func:
                    continue
                
                # Extract function name
                name_match = re.search(r'(\w+)\s*\(', func)
                func_name = name_match.group(1) if name_match else f"snippet_{i}"
                
                # Check token count and truncate if necessary
                token_count = self.count_tokens(func)
                if token_count > self.max_tokens:
                    func = self.truncate_to_token_limit(func)
                    print(f"Truncated function {func_name} in {file_path} from {token_count} tokens to {self.max_tokens} tokens")
                
                snippets.append({
                    'file': file_path,
                    'name': func_name,
                    'code': func
                })
                
        # For other file types, split into logical chunks
        else:
            lines = content.split('\n')
            chunk_size = 20  # Lines per chunk
            for i in range(0, len(lines), chunk_size):
                chunk = '\n'.join(lines[i:i+chunk_size])
                if chunk.strip():
                    # Check token count and truncate if necessary
                    token_count = self.count_tokens(chunk)
                    if token_count > self.max_tokens:
                        chunk = self.truncate_to_token_limit(chunk)
                        print(f"Truncated chunk in {file_path} from {token_count} tokens to {self.max_tokens} tokens")
                        
                    snippets.append({
                        'file': file_path,
                        'name': f"snippet_{i//chunk_size}",
                        'code': chunk
                    })
        
        return snippets
    
    def build_index(self, extensions=None, force_rebuild=False, progress_callback=None):
        """
        Build the search index by extracting code snippets and generating embeddings.
        
        Args:
            extensions (list): List of file extensions to include
            force_rebuild (bool): Force rebuilding the index even if it exists
            progress_callback (function): Callback for progress updates (0-100)
        """
        if os.path.exists(self.db_file) and not force_rebuild:
            print(f"Loading existing index from {self.db_file}")
            with open(self.db_file, 'r') as f:
                data = json.load(f)
                self.code_snippets = data['snippets']
                self.embeddings = np.array(data['embeddings'])
            print(f"Loaded {len(self.code_snippets)} snippets")
            if progress_callback:
                progress_callback(100)
            return
        
        if extensions is None:
            extensions = ['.c', '.h', '.cpp', '.hpp', '.py', '.js', '.sh']
        
        # Find all source files
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(self.source_dir, f"**/*{ext}"), recursive=True))
        
        print(f"Found {len(all_files)} source files")
        if progress_callback:
            progress_callback(5)  # 5% progress after finding files
        
        # Extract snippets from each file
        for i, file_path in enumerate(all_files):
            snippets = self.extract_snippets(file_path)
            self.code_snippets.extend(snippets)
            
            # Update progress (file processing phase: 5% to 50%)
            if progress_callback and len(all_files) > 0:
                progress = 5 + (i / len(all_files)) * 45
                progress_callback(int(progress))
        
        print(f"Extracted {len(self.code_snippets)} code snippets")
        
        # Generate embeddings for all snippets
        self.embeddings = self.get_embeddings(
            [s['code'] for s in self.code_snippets], 
            progress_callback=progress_callback
        )
        
        # Save to disk
        with open(self.db_file, 'w') as f:
            json.dump({
                'snippets': self.code_snippets,
                'embeddings': self.embeddings.tolist()
            }, f)
        
        print(f"Built and saved index with {len(self.code_snippets)} snippets")
        if progress_callback:
            progress_callback(100)  # Complete
    
    def get_embeddings(self, texts, progress_callback=None):
        """
        Get embeddings for a list of texts using OpenAI API.
        
        Args:
            texts (list): List of texts to embed
            progress_callback (function): Callback for progress updates
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        embeddings = []
        batch_size = 100  # Process in batches to avoid API limits
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_end = min(i+batch_size, len(texts))
            print(f"Getting embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({i}-{batch_end})")
            
            # Update progress (embedding phase: 50% to 95%)
            if progress_callback:
                progress = 50 + (i / len(texts)) * 45
                progress_callback(int(progress))
                
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
                # If a batch fails, try processing one at a time
                for j, text in enumerate(batch):
                    try:
                        # Further truncate if necessary
                        if self.count_tokens(text) > 8000:
                            text = self.truncate_to_token_limit(text, 8000)
                            
                        response = self.client.embeddings.create(
                            model=self.model,
                            input=[text]
                        )
                        embeddings.append(response.data[0].embedding)
                        print(f"  Processed item {i+j+1}/{len(texts)} individually")
                    except Exception as inner_e:
                        print(f"  Failed to process item {i+j+1}/{len(texts)}: {inner_e}")
                        # Add a zero vector as placeholder for failed embeddings
                        # Use same dimensionality as successful embeddings or a default size
                        dim = 1536  # Default for text-embedding-3-small
                        if embeddings:
                            dim = len(embeddings[0])
                        embeddings.append(np.zeros(dim).tolist())
        
        return np.array(embeddings)
    
    def search(self, query, top_k=5):
        """
        Search for code snippets matching the query.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            list: List of matching code snippets
        """
        if not self.code_snippets:
            print("No index built. Run build_index() first.")
            return []
        
        # Get embedding for the query
        query_embedding = self.get_embeddings([query])[0]
        
        # Calculate similarity scores
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            snippet = self.code_snippets[idx]
            results.append({
                'file': snippet['file'],
                'name': snippet['name'],
                'code': snippet['code'],
                'similarity': similarities[idx]
            })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Semantic Code Search')
    parser.add_argument('--source_dir', type=str, required=True, help='Source code directory')
    parser.add_argument('--token', type=str, required=True, help='OpenAI API token')
    parser.add_argument('--build', action='store_true', help='Build the search index')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--top_k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    search = SemanticCodeSearch(args.source_dir, args.token)
    
    if args.build:
        search.build_index(force_rebuild=True)
    
    if args.query:
        results = search.search(args.query, args.top_k)
        print(f"\nFound {len(results)} results for query: {args.query}\n")
        
        for i, result in enumerate(results):
            print(f"Result {i+1} (similarity: {result['similarity']:.4f})")
            print(f"File: {result['file']}")
            print(f"Function: {result['name']}")
            print("Code:")
            print("-" * 80)
            print(result['code'])
            print("=" * 80)
            print()

if __name__ == "__main__":
    main()
