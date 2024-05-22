import torch
import clip
import clip_helper
from clip_helper_clean import load as load_clean
from clip_model_clean import CLIP as CLIPClean
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_helper.load("ViT-B/32", device=device)
model2 = clip.load_clip()

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

model_clean, preprocess_clean = load_clean("ViT-B/32", device=device)
image_clean = preprocess_clean(Image.open("CLIP.png")).unsqueeze(0).to(device)

model_clean_loaded = CLIPClean()

with open("clip_simplified.pt", "rb") as opened_file:
    state_dict = torch.load(opened_file, map_location="cpu")

model_clean_loaded.load_state_dict(state_dict, strict=False)

with torch.no_grad():
    image_features = model.encode_image(image)

    image_features_my = model2.encode_image(image)

    image_features_clean = model_clean.encode_image(image)

    torch.save(model_clean.state_dict(), "clip_simplified.pt")
    torch.save(model.state_dict(), "clip_original.pt")

    image_features_clean_loaded = model_clean_loaded.encode_image(image)

    print("hi")
    # text_features = model.encode_text(text)
    #
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
