import os
import json
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from gpt import GPT, GPTConfig, transpose_specific_layers
from transformers import GPT2Tokenizer
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

    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 5e-5
    STEP_SIZE = 1
    GAMMA = 0.9
    CLIP_GRAD_NORM = 1.0  # Gradient clipping
    GRADIENT_ACCUMULATION_STEPS = 4  # Gradient accumulation

    state_dict = torch.load("gpt2.bin", map_location="cpu")
    state_dict_transposed = transpose_specific_layers(state_dict)

    model.load_state_dict(state_dict_transposed, strict=False)
    transform = _transform(224)
    coco_dataset = COCODataset(coco_root_dir, coco_ann_file, transform=transform)
    coco_dataloader = DataLoader(
        coco_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    vision_encoder = load_clip()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("done loading tokenizer")

    # Freeze the weights of the model and vision encoder
    for param in model.parameters():
        param.requires_grad = False

    for param in vision_encoder.parameters():
        param.requires_grad = False

    connector = VisionLanguageConnector()
    optimizer = optim.AdamW(connector.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    model.eval()  # Set the model to evaluation mode since it's frozen
    vision_encoder.eval()  # Set the vision encoder to evaluation mode since it's frozen
    connector.train()  # Set the connector to training mode

    for epoch in range(EPOCHS):
        epoch_loss = 0
        optimizer.zero_grad()

        for i, (images, captions) in enumerate(coco_dataloader):
            image_features = vision_encoder.encode_image(images)
            vision_embed = connector(image_features)

            # Assuming you have these initialized somewhere
            num_patches = vision_embed.size(1)
            embed_dim = vision_embed.size(2)

            # Special tokens (assumed to be defined in your tokenizer's vocabulary)
            image_end_token_id = tokenizer.convert_tokens_to_ids("[IMG_END]")
            end_text_token_id = tokenizer.eos_token_id
            pad_token_id = 0

            # Step 1: Tokenize captions
            tokenized_captions = [
                torch.tensor(tokenizer.encode(caption)) for caption in captions
            ]

            x_train = [
                torch.cat([torch.tensor([image_end_token_id]), tokenized_captions[i]])
                for i in range(BATCH_SIZE)
            ]

            # Step 4: Pad the training examples to the max length in the batch
            max_length = max([seq.size(0) for seq in x_train])
            x_train_padded = torch.full(
                (BATCH_SIZE, max_length), pad_token_id, dtype=torch.long
            )

            for i, seq in enumerate(x_train):
                x_train_padded[i, : seq.size(0)] = seq

            # Step 5: Create target tensor with shifted captions and masked image embeddings
            y_train = torch.full((BATCH_SIZE, x_train_padded.size(1)), -100)

            # Fill in y_train with appropriately shifted sequences
            for i in range(BATCH_SIZE):
                # Get the current sequence and calculate the needed shift
                current_sequence = tokenized_captions[i]
                shifted_sequence = torch.cat(
                    [
                        current_sequence,
                        torch.tensor([end_text_token_id], dtype=torch.long),
                    ]
                )

                # Fill the y_train tensor for the current batch index
                y_train[i, : shifted_sequence.size(0)] = shifted_sequence

            vision_embed_mask = torch.full((BATCH_SIZE, num_patches), -100)

            y_train = torch.cat([vision_embed_mask, y_train], dim=1)

            padding_mask = (x_train_padded != pad_token_id).long()
            prefix_padding_mask = torch.full((BATCH_SIZE, num_patches), 1)
            padding_mask = torch.cat([prefix_padding_mask, padding_mask], dim=1)

            logits, loss = model(
                x_train_padded, vision_embed, targets=y_train, padding_mask=padding_mask
            )

            # Scale the loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()  # Backpropagate the loss

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    connector.parameters(), CLIP_GRAD_NORM
                )  # Gradient clipping
                optimizer.step()  # Update the model parameters
                optimizer.zero_grad()
                scheduler.step()  # Update the learning rate

            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            print(
                f"Batch {i + 1}/{len(coco_dataloader)} - Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS}"
            )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - Average Loss: {epoch_loss / len(coco_dataloader)}"
        )

    print("Training complete.")
