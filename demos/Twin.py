import json
import torch
from cluster_determination import classPredict
from transformers import RobertaTokenizer, RobertaModel, T5ForConditionalGeneration, T5EncoderModel

#garden
query = "When in the gardenRoutine, the system starts watering the garden, sets the watering schedule to 06:00, and adjusts the water flow based on soil moisture and weather forecast. If the forecast is 'rain', sets water flow to level 1; if soil moisture is below 30, sets water flow to level 5; otherwise, sets water flow to level 3. When in the eveningGardenRoutine, the system stops watering and sets water flow to level 2."

def Twin():
    description = query
    predicted_classes = classPredict(description)
    search_files = [f"dataSet/cluster/{predicted_class}.json" for predicted_class in predicted_classes]


    # Load the corresponding model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("fine_tuned_codet5")
    encoder_model = T5EncoderModel.from_pretrained("fine_tuned_codet5") # local_codet5_base, fine_tuned_codet5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model.to(device)

    # Load code data from multiple JSON files
    code_data = []
    for search_file in search_files:
        with open(search_file, 'r', encoding='utf-8') as f:
            code_data.extend(json.load(f))

    query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    query_embedding = encoder_model(**query_inputs).last_hidden_state.mean(dim=1)

    similarities = []
    for item in code_data:
        code = item["code"]
        code_inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True).to(device)
        code_embedding = encoder_model(**code_inputs).last_hidden_state.mean(dim=1)

        # Calculate cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(query_embedding, code_embedding, dim=-1)

        similarities.append((cosine_sim.item(), item["code"], item["template_docstring"]))

    # Sort by similarity
    similarities.sort(reverse=True, key=lambda x: x[0])

    # Set successRate@K
    top_K_similarities = similarities[:1]  # K
    for sim, code, docstring in top_K_similarities:
        print(f"{code}")

def main():
    Twin()


if __name__ == "__main__":
    main()
