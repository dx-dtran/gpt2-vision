from clip_original import load
import numpy as np
import torch
from tqdm.notebook import tqdm
from imagenetv2_pytorch import ImageNetV2Dataset

if __name__ == "__main__":

    model, preprocess = load("ViT-B/32")
    input_resolution = model.visual.input_resolution

    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )
    print("Input resolution:", input_resolution)

    images = ImageNetV2Dataset(transform=preprocess)
    loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)

    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for i, (images, target) in enumerate(tqdm(loader)):
            # images = images
            # target = target

            # predict
            image_features = model.encode_image(images)
            image_features_original = model.encode_image_original(images)
            if i > 4:
                break

            print(image_features)
            print(image_features.shape)

            print(image_features_original)
            print(image_features_original.shape)
