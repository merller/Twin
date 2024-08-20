import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel
from sklearn.metrics.pairwise import cosine_similarity
import esprima
from collections import defaultdict

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to('cuda') if 't5' in model_name else AutoModel.from_pretrained(model_name).to('cuda')
    return tokenizer, model

# Initialize model and tokenizer
model_name = "dataSet/local_codet5_base"  
tokenizer, model = load_model_and_tokenizer(model_name)

class FunctionCallGraph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_function(self, function_name):
        if function_name not in self.graph:
            self.graph[function_name] = []

    def add_function_call(self, caller, callee):
        if caller is not None:
            self.graph[caller].append(callee)
    
    def get_pruned_graph(self):
        pruned_graph = {func: calls for func, calls in self.graph.items() if calls}
        def expand_calls(function, visited=None):
            if visited is None:
                visited = set()
            expanded_calls = []
            for callee in pruned_graph.get(function, []):
                if callee not in visited:
                    visited.add(callee)
                    if callee in pruned_graph:
                        expanded_calls.extend(expand_calls(callee, visited))
                    else:
                        expanded_calls.append(callee)
            return expanded_calls

        expanded_graph = {}
        for func in pruned_graph:
            expanded_graph[func] = expand_calls(func)
        return expanded_graph
    
    def to_text(self):
        expanded_graph = self.get_pruned_graph()
        text_representation = []
        for func, calls in expanded_graph.items():
            calls_str = ', '.join(calls)
            text_representation.append(f"{func} calls {calls_str}")
        return ' '.join(text_representation)

def build_function_call_graph(code):
    call_graph = FunctionCallGraph()
    ast = esprima.parseScript(code)
    def traverse(node):
        if isinstance(node, list):
            for child in node:
                traverse(child)
            return
        
        if isinstance(node, esprima.nodes.FunctionDeclaration):
            function_name = node.id.name
            call_graph.add_function(function_name)
            call_graph.current_function = function_name
        
        elif isinstance(node, esprima.nodes.CallExpression):
            if isinstance(node.callee, esprima.nodes.Identifier):
                callee_name = node.callee.name
                call_graph.add_function_call(call_graph.current_function, callee_name)

        for field_name, field_value in node.__dict__.items():
            if isinstance(field_value, (esprima.nodes.Node, list)):
                traverse(field_value)
    
    traverse(ast.body)
    return call_graph.to_text()

# Convert AST to text format
def ast_to_text(code):
    ast = esprima.parseScript(code)
    node_types = []
    def visit_node(node):
        if isinstance(node, dict):
            node_type = node.get('type', None)
            if node_type:
                node_types.append(node_type)
            for key, value in node.items():
                visit_node(value)
        elif isinstance(node, list):
            for item in node:
                visit_node(item)
    visit_node(ast)
    return ' '.join(node_types)

# Extract function names and event keywords
def extract_function_and_event_names(code):
    function_names = re.findall(r'\bdef (\w+)\(', code)
    event_names = re.findall(r'\bevent\.(\w+)\(', code)
    return function_names + event_names

# Encode code using extracted function and event names
def encode(code):
    words = extract_function_and_event_names(code)
    ast = ast_to_text(code)
    icfg = build_function_call_graph(code)

    word_embed = encode_words(words)
    ast_embed = encode_text(ast)
    icfg_embed = encode_text(icfg)

    # Load the saved Query, Key, and Value matrices
    weight_file='models/attention/attention.pth'
    saved_weights = torch.load(weight_file)
    query_matrix = saved_weights['query_matrix']
    key_matrix = saved_weights['key_matrix']
    value_matrix = saved_weights['value_matrix']
    
    encodings = torch.stack([word_embed, ast_embed, icfg_embed], dim=0)

    # Compute the Query, Key, and Value outputs
    query_result = torch.matmul(encodings, query_matrix.T)  
    key_result = torch.matmul(encodings, key_matrix.T)      
    value_result = torch.matmul(encodings, value_matrix.T)  

    # Compute the attention scores (Query @ Key.T)
    attention_scores = torch.matmul(query_result, key_result.transpose(-2, -1)) / torch.sqrt(torch.tensor(query_result.size(-1), dtype=torch.float))

    # Apply softmax to normalize the attention scores and get attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # Apply attention weights to the Value and compute the weighted sum
    weighted_encoding = torch.sum(attention_weights.unsqueeze(-1) * value_result, dim=0)

    return weighted_encoding
    

# Encode words and move tensor to GPU
def encode_words(words):
    if words:
        inputs = tokenizer('[SEP]'.join(words), return_tensors='pt', padding=True, truncation=True).to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)  # Mean as representation, remove extra dimension
    else:
        return torch.zeros(768).to('cuda')  # Return zero vector to prevent empty input

# Encode text and move tensor to GPU
def encode_text(text):
    inputs = tokenizer.encode_plus(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0)  # Mean as representation, remove extra dimension

# Read JSON data
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Extract code content from JSON data
def extract_code(data):
    return [item['code'] for item in data]

# Compute average encoding for each class in folder
def compute_class_average_encodings(folder_path):
    class_avg_encodings = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            data = read_json(file_path)
            code_list = extract_code(data)
            encodings = [encode(code) for code in code_list]
            avg_encoding = torch.stack(encodings).mean(dim=0)
            class_avg_encodings.append((filename, avg_encoding))
    return class_avg_encodings

# Calculate similarity and return top 1 class
def get_top_1_similar_classes(class_avg_encodings, code_embedding):
    similarities = [(filename, cosine_similarity(code_embedding.cpu().unsqueeze(0), class_encoding.cpu().unsqueeze(0)).item()) 
                    for filename, class_encoding in class_avg_encodings]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:1]

# Main function
def main(folder_path, scene_cluster_file):
    class_avg_encodings = compute_class_average_encodings(folder_path)
    scene_data = read_json(scene_cluster_file)
    top_1_classes_per_code = {}
    for idx, item in enumerate(scene_data):
        code = item.get('code', '')
        if code:
            code_embedding = encode(code)
            top_1_classes = get_top_1_similar_classes(class_avg_encodings, code_embedding)
            top_1_classes_per_code[idx] = top_1_classes[0][0]  # Only take filename
    return top_1_classes_per_code

# Usage example
folder_path = 'dataSet/scene/cluster'
scene_cluster_file = 'dataSet/scene/Scene_clusterMetric.json'
top_1_classes_per_code = main(folder_path, scene_cluster_file)
with open(scene_cluster_file, 'r', encoding='utf-8') as f:
    scene_data = json.load(f)

# Iterate over top_1_classes_per_code and save objects to corresponding files
for idx in top_1_classes_per_code:
    print(file_name)

