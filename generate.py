import os
import torch
import random
import time
import matplotlib.pyplot as plt
from PIL import Image
from transformers import GPT2Tokenizer
from clip import load_clip
from vision_language_connector import VisionLanguageConnector
from gpt import GPT, GPTConfig, transpose_specific_layers, generate_text


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


def process_images_and_generate_text(
    image_folder,
    output_folder,
    vision_encoder,
    preprocess,
    model,
    connector,
    tokenizer,
    max_generations=20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_encoder.to(device)
    model.to(device)
    connector.to(device)

    model.eval()
    vision_encoder.eval()
    connector.eval()

    os.makedirs(output_folder, exist_ok=True)

    image_files = os.listdir(image_folder)
    random.shuffle(image_files)

    for i, image_filename in enumerate(image_files):
        if i > max_generations:
            break

        image_path = os.path.join(image_folder, image_filename)
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = vision_encoder.encode_image(image_tensor)
            image_features = image_features.to(device).to(
                next(connector.parameters()).dtype
            )
            vision_embed = connector(image_features)
            vision_embed = vision_embed.to(device).to(
                next(connector.parameters()).dtype
            )

        generated_text = generate_text(
            model, tokenizer, vision_embeds=vision_embed[0], temperature=0.2
        )

        save_image_and_caption_to_png(
            output_folder, image, generated_text, image_filename
        )


def save_image_and_caption_to_png(folder, image, caption, image_filename):
    fig, (ax_image, ax_caption) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [10, 1]},
    )

    ax_image.imshow(image)
    ax_image.axis("off")

    ax_image.set_aspect("equal", adjustable="box")

    ax_caption.text(0.5, 0.5, caption, ha="center", va="center", wrap=True, fontsize=12)
    ax_caption.axis("off")

    png_filename = os.path.join(folder, f"{os.path.splitext(image_filename)[0]}.png")
    plt.savefig(png_filename, bbox_inches="tight", pad_inches=0.1)
    plt.close()


if __name__ == "__main__":
    current_time = time.time()
    current_time = time.strftime("%Y-%m-%d-%H%M%S", time.localtime(current_time))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_folder = "input_images"
    output_folder = f"outputs-{current_time}"

    gpt_weights_path = "gpt2.pt"
    connector_weights_path = "connector_weights_3000_0.pt"

    vision_encoder, preprocess, model, connector, tokenizer = load_models_and_tokenizer(
        gpt_weights_path, connector_weights_path, device
    )

    process_images_and_generate_text(
        image_folder,
        output_folder,
        vision_encoder,
        preprocess,
        model,
        connector,
        tokenizer,
    )
