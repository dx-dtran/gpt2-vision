import os
import sys
import json
import gc
import time
import logging
import matplotlib.pyplot as plt
import pytz
import torch
import torch.optim as optim
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer
from PIL import Image
from gpt import GPT, GPTConfig, transpose_specific_layers, generate_text
from clip import load_clip
from vision_language_connector import VisionLanguageConnector

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def save_image_and_caption_to_png(folder, image_tensor, caption, iteration):
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Iteration {iteration}: {caption}", wrap=True)
    png_filename = os.path.join(folder, f"iteration_{iteration}.png")
    plt.savefig(png_filename)
    plt.close()


def save_connector_weights(connector, folder, iteration):
    filename = os.path.join(folder, f"connector_weights_{iteration}.pt")
    torch.save(connector.state_dict(), filename)


def setup_logger():
    pacific_time = pytz.timezone("US/Pacific")
    timestamp = datetime.now(pacific_time).strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"training_log_{timestamp}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_filename


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, message):
        if message != "\n":
            self.buffer += message

    def flush(self):
        if self.buffer:
            self.logger.log(self.level, self.buffer.strip())
            self.buffer = ""


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


def load_model_and_tokenizer():
    config = GPTConfig()
    model = GPT(config)
    state_dict = torch.load("gpt2.pt", map_location="cpu")
    state_dict_transposed = transpose_specific_layers(state_dict)
    model.load_state_dict(state_dict_transposed, strict=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def prepare_training_components(learning_rate, weight_decay, t_max):
    vision_encoder, preprocess = load_clip()
    connector = VisionLanguageConnector()
    optimizer = optim.AdamW(
        connector.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
    return vision_encoder, preprocess, connector, optimizer, scheduler


def tokenize_and_prepare_batches(captions, tokenizer, batch_size, num_patches):
    image_end_token_id = tokenizer.bos_token_id
    end_text_token_id = tokenizer.eos_token_id
    pad_token_id = 0

    tokenized_captions = [
        torch.tensor(tokenizer.encode(caption)).detach() for caption in captions
    ]
    x_train = [
        torch.cat([torch.tensor([image_end_token_id]), tokenized_captions[i]]).detach()
        for i in range(batch_size)
    ]
    max_length = max([seq.size(0) for seq in x_train])
    x_train_padded = torch.full(
        (batch_size, max_length), pad_token_id, dtype=torch.long
    ).detach()
    for i, seq in enumerate(x_train):
        x_train_padded[i, : seq.size(0)] = seq

    y_train = torch.full((batch_size, x_train_padded.size(1)), -100).detach()
    for i in range(batch_size):
        current_sequence = tokenized_captions[i]
        shifted_sequence = torch.cat(
            [
                current_sequence,
                torch.tensor([end_text_token_id], dtype=torch.long).detach(),
            ]
        ).detach()
        y_train[i, : shifted_sequence.size(0)] = shifted_sequence

    vision_embed_mask = torch.full((batch_size, num_patches), -100).detach()
    y_train = torch.cat([vision_embed_mask, y_train], dim=1).detach()

    padding_mask = (x_train_padded != pad_token_id).long().detach()
    prefix_padding_mask = torch.full((batch_size, num_patches), 1).detach()
    padding_mask = torch.cat([prefix_padding_mask, padding_mask], dim=1).detach()

    del tokenized_captions, x_train, current_sequence, shifted_sequence
    gc.collect()
    return x_train_padded, y_train, padding_mask


def train_model(
    model,
    connector,
    vision_encoder,
    tokenizer,
    data_loader,
    val_data_loader,
    optimizer,
    scheduler,
    epochs,
    batch_size,
    gradient_accumulation_steps,
    max_grad_norm,
    device,
):
    model.to(device)
    vision_encoder.to(device)
    connector.to(device)

    model.eval()
    vision_encoder.eval()
    connector.train()

    pacific_time = pytz.timezone("US/Pacific")
    timestamp = datetime.now(pacific_time).strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = f"training_results_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    weights_output_folder = f"weights_results_{timestamp}"
    os.makedirs(weights_output_folder, exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()

        for i, (images, captions) in enumerate(data_loader):
            start_time = time.time()

            images = images.to(device)

            image_features = vision_encoder.encode_image(images)
            image_features = image_features.to(device).to(
                next(connector.parameters()).dtype
            )
            vision_embed = connector(image_features)
            vision_embed = vision_embed.to(device).to(
                next(connector.parameters()).dtype
            )
            num_patches = vision_embed.size(1)

            if len(captions) != batch_size:
                logger.info(f"Skipping batch {i + 1} due to mismatched batch size.")
                del images, image_features, vision_embed
                torch.cuda.empty_cache()
                gc.collect()
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

            torch.nn.utils.clip_grad_norm_(connector.parameters(), max_grad_norm)

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * gradient_accumulation_steps

            end_time = time.time()
            iteration_time = end_time - start_time

            logger.info(
                f"Batch {i + 1}/{len(data_loader)} - Loss: {loss.item() * gradient_accumulation_steps:.4f} - LR: {scheduler.get_last_lr()[0]:.9f} - Iter: {iteration_time:.4f} sec"
            )

            if (i + 1) % 100 == 0 or (i + 1) == 1:
                generated_text = generate_text(
                    model, tokenizer, vision_embeds=vision_embed[0]
                )
                save_image_and_caption_to_png(
                    output_folder, images[0], generated_text, i + 1
                )

                validate_model(
                    model,
                    connector,
                    vision_encoder,
                    tokenizer,
                    val_data_loader,
                    batch_size,
                    device,
                )

            if (i + 1) % 1000 == 0:
                save_connector_weights(connector, weights_output_folder, i + 1)

            del (
                images,
                image_features,
                vision_embed,
                x_train_padded,
                y_train,
                padding_mask,
            )
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - Average Loss: {epoch_loss / len(data_loader):.6f}"
        )
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Training complete.")


def validate_model(
    model, connector, vision_encoder, tokenizer, validation_loader, batch_size, device
):
    model.eval()
    connector.eval()
    vision_encoder.eval()

    total_val_loss = 0

    with torch.no_grad():
        for batch_num, (images, captions) in enumerate(validation_loader):

            if batch_num > 20:
                break

            images = images.to(device)

            image_features = vision_encoder.encode_image(images)
            image_features = image_features.to(device).to(
                next(connector.parameters()).dtype
            )
            vision_embed = connector(image_features)
            vision_embed = vision_embed.to(device).to(
                next(connector.parameters()).dtype
            )
            num_patches = vision_embed.size(1)

            if len(captions) != batch_size:
                del images, image_features, vision_embed
                torch.cuda.empty_cache()
                gc.collect()
                continue

            x_val_padded, y_val, padding_mask = tokenize_and_prepare_batches(
                captions, tokenizer, batch_size, num_patches
            )

            x_val_padded = x_val_padded.to(device)
            y_val = y_val.to(device)
            padding_mask = padding_mask.to(device)

            logits, val_loss = model(
                x_val_padded, vision_embed, targets=y_val, padding_mask=padding_mask
            )

            total_val_loss += val_loss.item()
            logger.info(f"Validation Batch Loss: {val_loss.item():.4f}")

            del (
                images,
                image_features,
                vision_embed,
                x_val_padded,
                y_val,
                padding_mask,
            )
            torch.cuda.empty_cache()
            gc.collect()

    average_val_loss = total_val_loss / len(validation_loader)
    logger.info(f"Validation Loss Estimate: {average_val_loss:.4f}")

    return average_val_loss


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    CLIP_GRAD_NORM = 1.0
    GRADIENT_ACCUMULATION_STEPS = 1
    coco_root_dir = "../coco/train2017"
    coco_ann_file = "../coco/annotations/captions_train2017.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_encoder, preprocess, connector, optimizer, scheduler = (
        prepare_training_components(LEARNING_RATE, WEIGHT_DECAY, 18000 * EPOCHS)
    )
    freeze_model_parameters(vision_encoder)

    coco_dataset = COCODataset(coco_root_dir, coco_ann_file, transform=preprocess)
    train_size = int(0.8 * len(coco_dataset))
    val_size = len(coco_dataset) - train_size
    train_dataset, val_dataset = random_split(coco_dataset, [train_size, val_size])

    coco_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )

    model, tokenizer = load_model_and_tokenizer()
    freeze_model_parameters(model)

    logger, log_filename = setup_logger()
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)

    train_model(
        model,
        connector,
        vision_encoder,
        tokenizer,
        coco_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        EPOCHS,
        BATCH_SIZE,
        GRADIENT_ACCUMULATION_STEPS,
        CLIP_GRAD_NORM,
        device,
    )

    logger.info(f"Log file: {log_filename}")
