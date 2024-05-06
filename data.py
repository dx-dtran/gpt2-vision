import os
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
    Compose,
)
import matplotlib.pyplot as plt


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

    transform = _transform(224)
    coco_dataset = COCODataset(coco_root_dir, coco_ann_file, transform=transform)
    coco_dataloader = DataLoader(
        coco_dataset, batch_size=8, shuffle=True, num_workers=4
    )

    # Test loading some samples
    for i, (image, captions) in enumerate(coco_dataloader):
        print("Captions:")
        print(captions)
        print(f"Image Shape: {image.shape}")
        # show_images(image)

        break
