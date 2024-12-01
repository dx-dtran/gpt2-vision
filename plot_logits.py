import os
import sys
import json
import gc
import logging
import pytz
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer
from PIL import Image
from gpt import GPT, GPTConfig, transpose_specific_layers
from clip import load_clip
from vision_language_connector_bad import VisionLanguageConnectorBad
from vision_language_connector import VisionLanguageConnector

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


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


def load_model_and_tokenizer(device):
    config = GPTConfig()
    model = GPT(config)
    state_dict = torch.load("gpt2.pt", map_location="cpu")
    state_dict_transposed = transpose_specific_layers(state_dict)
    model.load_state_dict(state_dict_transposed, strict=False)
    model = model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer


def prepare_training_components(learning_rate, weight_decay):
    vision_encoder, preprocess = load_clip()
    optimizer = optim.AdamW(
        connector.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True, min_lr=1e-7
    )
    return vision_encoder, preprocess, connector, optimizer, scheduler


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


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


def validate_model(
    model,
    connector,
    vision_encoder,
    tokenizer,
    validation_loader,
    batch_size,
    device,
    max_batches=20,
):
    model.eval()
    connector.eval()
    vision_encoder.eval()

    total_val_loss = 0

    with torch.no_grad():
        for batch_num, (images, captions) in enumerate(validation_loader):

            if batch_num > max_batches:
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

            logits = logits.view(-1, logits.size(-1))
            y_val = y_val.view(-1)

            valid_mask = y_val != -100

            filtered_targets = y_val[valid_mask]
            num_logits_to_print = 10
            num_tokens = len(filtered_targets)
            cols = 5
            rows = math.ceil(num_tokens / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))

            axes = axes.flatten()

            for i, (target, logit_row) in enumerate(
                zip(filtered_targets, logits[valid_mask])
            ):
                token_text = tokenizer.decode([target.item()])
                ax = axes[i]

                softmax_probs = F.softmax(logit_row, dim=0).cpu().numpy()

                top_k_probs_indices = torch.topk(
                    torch.tensor(softmax_probs), k=num_logits_to_print
                )
                top_k_probs = top_k_probs_indices.values.numpy()
                top_k_indices = top_k_probs_indices.indices.numpy()

                ax.bar(range(num_logits_to_print), top_k_probs)
                ax.set_title(f"Token {i}: {token_text}")
                ax.set_xticks(range(num_logits_to_print))
                ax.set_xticklabels(
                    [tokenizer.decode([idx]) for idx in top_k_indices],
                    rotation=45,
                    ha="right",
                )
                ax.set_ylabel("Softmax Probability")
                ax.set_xlabel("Top Predictions")

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.show()

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

    average_val_loss = total_val_loss / max_batches
    logger.info(f"Validation Loss Estimate: {average_val_loss:.4f}")

    return average_val_loss


if __name__ == "__main__":
    BATCH_SIZE = 1
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    CLIP_GRAD_NORM = 1.0
    GRADIENT_ACCUMULATION_STEPS = 1
    coco_root_dir = "../coco/val2017"
    coco_ann_file = "../coco/annotations/captions_val2017.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    connector_weights_path = "vl_connector_large_mlp_collapsed.pt"
    # connector_weights_path = "vl_connector.pt"

    vision_encoder, preprocess = load_clip(device)
    connector = VisionLanguageConnectorBad()
    # connector = VisionLanguageConnector()
    connector.load_state_dict(torch.load(connector_weights_path, map_location="cpu"))

    connector = connector.to(device)

    freeze_model_parameters(vision_encoder)

    coco_dataset = COCODataset(coco_root_dir, coco_ann_file, transform=preprocess)
    train_size = int(0.9 * len(coco_dataset))
    val_size = len(coco_dataset) - train_size
    train_dataset, val_dataset = random_split(coco_dataset, [train_size, val_size])

    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )

    model, tokenizer = load_model_and_tokenizer(device)
    freeze_model_parameters(model)

    logger, log_filename = setup_logger()
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)
    val_loss = validate_model(
        model,
        connector,
        vision_encoder,
        tokenizer,
        val_dataloader,
        BATCH_SIZE,
        device,
    )

    logger.info(f"Log file: {log_filename}")
