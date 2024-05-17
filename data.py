import os
import json
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from PIL import Image
from torchvision.transforms import (
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
    Compose,
)
from gpt import GPT, GPTConfig, transpose_specific_layers, generate_text
from clip import load_clip
from vision_language_connector import VisionLanguageConnector


def convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(n_px):
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
        self.img_caption_pairs, self.id_to_filename = self.load_annotations(ann_file)

    def load_annotations(self, ann_file):
        with open(ann_file, "r") as f:
            data = json.load(f)
        img_caption_pairs = [
            (ann["image_id"], ann["caption"]) for ann in data["annotations"]
        ]
        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        return img_caption_pairs, id_to_filename

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
    for i in range(n_examples):
        image_tensor = batch_images[i].permute(1, 2, 0)
        image_array = image_tensor.cpu().numpy()
        ax = axes[i]
        ax.imshow(image_array)
        ax.set_title(f"Image {i + 1}")
        ax.axis("off")
    plt.show()


def show_image(image_tensor):
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def load_model_and_tokenizer():
    config = GPTConfig()
    model = GPT(config)
    state_dict = torch.load("gpt2.bin", map_location="cpu")
    state_dict_transposed = transpose_specific_layers(state_dict)
    model.load_state_dict(state_dict_transposed, strict=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def prepare_training_components(learning_rate, t_max):
    vision_encoder = load_clip()
    connector = VisionLanguageConnector()
    optimizer = optim.AdamW(connector.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
    return vision_encoder, connector, optimizer, scheduler


def tokenize_and_prepare_batches(captions, tokenizer, batch_size, num_patches):
    image_end_token_id = tokenizer.bos_token_id
    end_text_token_id = tokenizer.eos_token_id
    pad_token_id = 0

    tokenized_captions = [
        torch.tensor(tokenizer.encode(caption)) for caption in captions
    ]
    x_train = [
        torch.cat([torch.tensor([image_end_token_id]), tokenized_captions[i]])
        for i in range(batch_size)
    ]
    max_length = max([seq.size(0) for seq in x_train])
    x_train_padded = torch.full(
        (batch_size, max_length), pad_token_id, dtype=torch.long
    )
    for i, seq in enumerate(x_train):
        x_train_padded[i, : seq.size(0)] = seq

    y_train = torch.full((batch_size, x_train_padded.size(1)), -100)
    for i in range(batch_size):
        current_sequence = tokenized_captions[i]
        shifted_sequence = torch.cat(
            [current_sequence, torch.tensor([end_text_token_id], dtype=torch.long)]
        )
        y_train[i, : shifted_sequence.size(0)] = shifted_sequence

    vision_embed_mask = torch.full((batch_size, num_patches), -100)
    y_train = torch.cat([vision_embed_mask, y_train], dim=1)

    padding_mask = (x_train_padded != pad_token_id).long()
    prefix_padding_mask = torch.full((batch_size, num_patches), 1)
    padding_mask = torch.cat([prefix_padding_mask, padding_mask], dim=1)

    return x_train_padded, y_train, padding_mask


def train_model(
    model,
    connector,
    vision_encoder,
    tokenizer,
    data_loader,
    optimizer,
    scheduler,
    epochs,
    batch_size,
    gradient_accumulation_steps,
    device,
):
    model.to(device)
    vision_encoder.to(device)
    connector.to(device)

    model.train()
    vision_encoder.eval()
    connector.train()

    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()

        for i, (images, captions) in enumerate(data_loader):
            start_time = time.time()

            images = images.to(device)

            image_features = vision_encoder.encode_image(images)
            vision_embed = connector(image_features)
            num_patches = vision_embed.size(1)

            # Ensure captions list length matches the batch size
            if len(captions) != batch_size:
                print(f"Skipping batch {i + 1} due to mismatched batch size.")
                continue

            x_train_padded, y_train, padding_mask = tokenize_and_prepare_batches(
                captions, tokenizer, batch_size, num_patches
            )

            x_train_padded = x_train_padded.to(device)
            y_train = y_train.to(device)
            padding_mask = padding_mask.to(device)

            logits, loss = model(
                x_train_padded,
                vision_embed,
                targets=y_train,
                padding_mask=padding_mask,
            )
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * gradient_accumulation_steps

            end_time = time.time()
            iteration_time = end_time - start_time

            print(
                f"Batch {i + 1}/{len(data_loader)} - Loss: {loss.item() * gradient_accumulation_steps:.4f} - LR: {scheduler.get_last_lr()[0]:.9f} - Iter: {iteration_time:.4f} sec"
            )

            if (i + 1) % 100 == 0 or (i + 1) == 1:
                generate_text(model, tokenizer, vision_embeds=vision_embed[0])

        print(
            f"Epoch {epoch + 1}/{epochs} - Average Loss: {epoch_loss / len(data_loader):.6f}"
        )

    print("Training complete.")


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    T_MAX = 10
    CLIP_GRAD_NORM = 1.0
    GRADIENT_ACCUMULATION_STEPS = 4
    coco_root_dir = "../coco/val2017"
    coco_ann_file = "../coco/annotations/captions_val2017.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_transform(224)
    coco_dataset = COCODataset(coco_root_dir, coco_ann_file, transform=transform)
    coco_dataloader = DataLoader(
        coco_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    model, tokenizer = load_model_and_tokenizer()

    vision_encoder, connector, optimizer, scheduler = prepare_training_components(
        LEARNING_RATE, T_MAX
    )
    freeze_model_parameters(vision_encoder)

    train_model(
        model,
        connector,
        vision_encoder,
        tokenizer,
        coco_dataloader,
        optimizer,
        scheduler,
        EPOCHS,
        BATCH_SIZE,
        GRADIENT_ACCUMULATION_STEPS,
        device,
    )
