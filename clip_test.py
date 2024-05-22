import torch
import clip_helper
from clip_model_clean import load_clip
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_helper.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)


clip_final, preprocess_final = load_clip()

image_final = preprocess_final(Image.open("CLIP.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)

    image_features_final = clip_final.encode_image(image_final)

    print("hi")
    # text_features = model.encode_text(text)
    #
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
