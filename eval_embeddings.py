import os
import torch
import random
from PIL import Image
from transformers import GPT2Tokenizer
from clip import load_clip
from vision_language_connector import VisionLanguageConnector
from gpt import GPT, GPTConfig, transpose_specific_layers


def load_models_and_tokenizer(gpt_model_path, connector_weights_path, device):
    vision_encoder, preprocess = load_clip(device)

    config = GPTConfig()
    model = GPT(config)
    state_dict = torch.load(gpt_model_path, map_location="cpu")
    state_dict_transposed = transpose_specific_layers(state_dict)
    model.load_state_dict(state_dict_transposed, strict=False)
    model = model.to(device)

    connector = VisionLanguageConnector()
    connector.load_state_dict(torch.load(connector_weights_path, map_location="cpu"))
    connector = connector.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return vision_encoder, preprocess, model, connector, tokenizer


def process_images_and_map_embeddings(
    image_folder,
    vision_encoder,
    preprocess,
    connector,
    gpt_model,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_encoder.to(device)
    connector.to(device)
    gpt_model.to(device)

    vision_encoder.eval()
    connector.eval()
    gpt_model.eval()

    image_files = os.listdir(image_folder)
    random.shuffle(image_files)

    for image_filename in image_files:
        image_path = os.path.join(image_folder, image_filename)
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = vision_encoder.encode_image(image_tensor)
            image_features = image_features.to(device).to(
                next(connector.parameters()).dtype
            )
            vision_embed = connector(image_features)

            gpt2_embedding_matrix = gpt_model.wte.weight
            results = []
            for embedding in vision_embed[0]:
                similarity = torch.nn.functional.cosine_similarity(
                    embedding.unsqueeze(0), gpt2_embedding_matrix, dim=1
                )
                closest_idx = torch.argmax(similarity).item()
                results.append((closest_idx, similarity[closest_idx].item()))

            print(f"Image: {image_filename}")
            for idx, (token_idx, similarity) in enumerate(results):
                print(
                    f"  Visual Embedding {idx}: Closest GPT-2 Token = {token_idx}, Similarity = {similarity:.4f}"
                )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_folder = "../coco/val2017"

    gpt_weights_path = "gpt2.pt"
    connector_weights_path = "connector_weights_1_0.pt"

    vision_encoder, preprocess, gpt_model, connector, _ = load_models_and_tokenizer(
        gpt_weights_path, connector_weights_path, device
    )

    process_images_and_map_embeddings(
        image_folder,
        vision_encoder,
        preprocess,
        connector,
        gpt_model,
    )
