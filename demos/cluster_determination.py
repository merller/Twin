import json
import torch
import re
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from collections import Counter

class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('/bert-base-uncased/')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.fc(output)

def preprocess_text(text):
    stopwords = set([
        "a", "an", "the", "and", "but", "or", "on", "in", "with", "is", "was", "were", "be", 
        "to", "of", "that", "which", "at", "by", "for", "from", "as", "about", "if", "when", 
        "this", "these", "those", "then", "there", "here", "it", "its", "they", "them", 
        "he", "she", "we", "you", "your", "my", "mine", "ours", "his", "her", "hers", 
        "morning","afternoon","evening","WHEN","DO","IF","THEN","OTHERWISE","WHILE",
        "their", "theirs", "me", "him", "routine", "suggest", "s", "sets", "c", "level", "set", "otherwise",
        "are", "starts", "start", "end", "ends", "end", "return", "returns", "returns", "returning", "returning", "return", "returns"
    ])
    def split_compound_word(word):
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', word).lower().split()
    words = []
    for word in text.split():
        words.extend(split_compound_word(word))

    words = [word for word in words if word not in stopwords]
    word_freq = Counter(words)
    top_words = [word for word, freq in word_freq.most_common(4)]
    return ' '.join(top_words)

def predict_scene(description, model, tokenizer, label_encoder, device):
    description = preprocess_text(description)
    inputs = tokenizer(description, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    top_3_indices = probabilities.argsort()[0, -1:][::-1]
    return label_encoder.inverse_transform(top_3_indices)

def load_model(model_path, n_classes, device):
    model = BertClassifier(n_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def classPredict(description):
    all_labels = [
        "garden", "kitchen", "factory", "hotel", "room", "office", "patient", 
        "security", "store", "sports", "home entertainment", "pet", "library", 
        "warehouse", "museum", "traffic", "school", "coffee"
    ]

    n_classes = len(set(all_labels))

    tokenizer = BertTokenizer.from_pretrained('/bert-base-uncased/')
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model_save_path = 'FCmodel.pth'
    model = load_model(model_save_path, n_classes, device)

    predicted_scenes = predict_scene(description, model, tokenizer, label_encoder, device)
    return predicted_scenes
def main():
    # Input description
    description = (
        "When in the gardenRoutine, the system starts watering the garden, sets the watering schedule to 06:00, and adjusts the water flow based on soil moisture and weather forecast. If the forecast is 'rain', sets water flow to level 1; if soil moisture is below 30, sets water flow to level 5; otherwise, sets water flow to level 3. When in the eveningGardenRoutine, the system stops watering and sets water flow to level 2."
    )

    # Predict scene categories
    predicted_classes = classPredict(description)

    # Print predicted classes
    print("Predicted Classes:", predicted_classes)


if __name__ == "__main__":
    main()

