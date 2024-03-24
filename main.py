from clip_simplified import load
from clip_original import load as original_load
from image import transform_image
import numpy as np
import torch
from tqdm.notebook import tqdm
from imagenetv2_pytorch import ImageNetV2Dataset

if __name__ == "__main__":

    model = load()
    model_original, _ = original_load("ViT-B/32")
    input_resolution = model.visual.input_resolution

    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )
    print("Input resolution:", input_resolution)

    images = ImageNetV2Dataset(transform=transform_image(224))
    loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)

    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for i, (images, target) in enumerate(tqdm(loader)):
            # predict
            image_features = model.encode_image(images)

            if i > 4:
                break

            print("original")
            print(image_features)
            print(image_features.shape)
