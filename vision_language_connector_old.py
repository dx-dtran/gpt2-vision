import torch.nn as nn


class VisionLanguageConnectorOld(nn.Module):
    def __init__(self, vision_embed_dim=768, language_embed_dim=768):
        super(VisionLanguageConnectorOld, self).__init__()

        self.vision_embed_dim = vision_embed_dim
        hidden_dim = self.vision_embed_dim * 4

        self.layers = nn.Sequential(
            nn.Linear(self.vision_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, language_embed_dim),
        )

    def forward(self, x):
        batch_size, num_visual_tokens, vision_embed_dim = x.shape

        assert vision_embed_dim == self.vision_embed_dim
        # Reshape x to [-1, vision_embed_dim] to apply the linear transformation
        x = x.view(-1, vision_embed_dim)
        # Apply the MLP
        x = self.layers(x)
        # Reshape back to [batch_size, num_visual_tokens, output_dim]
        x = x.view(batch_size, num_visual_tokens, -1)
        return x
