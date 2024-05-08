import os
import json
import matplotlib.pyplot as plt
import tiktoken
import torch

from PIL import Image
from clip import load_clip
from vision_language_connector import VisionLanguageConnector
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
    Compose,
)

from gpt import GPT, GPTConfig, transpose_specific_layers


def convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class COCODataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load annotations
        with open(ann_file, "r") as f:
            data = json.load(f)

        # Build (image, caption) pairs
        self.img_caption_pairs = []
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            caption = ann["caption"]
            self.img_caption_pairs.append((img_id, caption))

        # Create a mapping for image IDs to file names
        self.id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    def __len__(self):
        return len(self.img_caption_pairs)

    def __getitem__(self, idx):
        img_id, caption = self.img_caption_pairs[idx]
        img_path = os.path.join(self.root_dir, self.id_to_filename[img_id])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, caption


def show_images(batch_images):
    n_examples = batch_images.shape[0]

    fig, axes = plt.subplots(nrows=1, ncols=n_examples, figsize=(20, 5))
    # Loop through all images in the batch
    for i in range(n_examples):
        # Select one image from the batch
        image_tensor = batch_images[i]

        # Convert to (height, width, channels) for plotting
        image_tensor = image_tensor.permute(1, 2, 0)

        # If the tensor is on GPU, first move it to CPU and convert to numpy
        image_array = image_tensor.cpu().numpy()

        # Display the image
        ax = axes[i]
        ax.imshow(image_array)
        ax.set_title(f"Image {i + 1}")
        ax.axis("off")  # Hide axes

    plt.show()


if __name__ == "__main__":

    # Usage example
    coco_root_dir = "../coco/val2017"
    coco_ann_file = "../coco/annotations/captions_val2017.json"
    config = GPTConfig()
    model = GPT(config)

    state_dict = torch.load("gpt2.bin", map_location="cpu")
    state_dict_transposed = transpose_specific_layers(state_dict)

    model.load_state_dict(state_dict_transposed, strict=False)
    transform = _transform(224)
    coco_dataset = COCODataset(coco_root_dir, coco_ann_file, transform=transform)
    coco_dataloader = DataLoader(
        coco_dataset, batch_size=8, shuffle=True, num_workers=4
    )
    vision_encoder = load_clip()
    # Test loading some samples
    for i, (images, captions) in enumerate(coco_dataloader):
        print("Captions:")
        print(captions)
        print(f"Image Shape: {images.shape}")

        image_features = vision_encoder.encode_image(images)
        connector = VisionLanguageConnector()
        vision_embed = connector(image_features)

        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        start_ids = [encode(caption) for caption in captions]
        padded_lists = [lst + [0] * (1024 - len(lst) - 49) for lst in start_ids]
        x = torch.tensor(padded_lists)

        model(x, vision_embed)

        print(vision_embed.shape)
        print("done")

        break
