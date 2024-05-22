import torch
import clip
import clip_helper
from clip_helper_clean import load as load_clean
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_helper.load("ViT-B/32", device=device)
model2 = clip.load_clip()

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

model_clean, preprocess_clean = load_clean("ViT-B/32", device=device)
image_clean = preprocess_clean(Image.open("CLIP.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)

    image_features_my = model2.encode_image(image)

    image_features_clean = model_clean.encode_image(image)

    torch.save(model_clean.state_dict(), "clip_simplified.pt")

    print("hi")
    # text_features = model.encode_text(text)
    #
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
