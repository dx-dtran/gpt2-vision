import torch.nn as nn


class VisionLanguageConnectorResidual(nn.Module):
    def __init__(
        self,
        vision_embed_dim=768,
        language_embed_dim=768,
        hidden_multiplier=2,
        num_layers=2,
        dropout_rate=0.1,
    ):
        super(VisionLanguageConnectorResidual, self).__init__()

        self.vision_embed_dim = vision_embed_dim
        hidden_dim = self.vision_embed_dim * hidden_multiplier

        self.layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()

        for i in range(num_layers):
            input_dim = vision_embed_dim if i == 0 else hidden_dim
            output_dim = language_embed_dim if i == (num_layers - 1) else hidden_dim

            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, output_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                )
            )

            if input_dim != output_dim:
                self.residual_layers.append(nn.Linear(input_dim, output_dim))
            else:
                self.residual_layers.append(nn.Identity())

    def forward(self, x):
        batch_size, num_visual_tokens, vision_embed_dim = x.shape

        assert (
            vision_embed_dim == self.vision_embed_dim
        ), f"Expected vision_embed_dim={self.vision_embed_dim}, got {vision_embed_dim}"

        x = x.view(-1, vision_embed_dim)

        for layer, residual in zip(self.layers, self.residual_layers):
            residual_x = residual(x)
            x = layer(x) + residual_x

        x = x.view(batch_size, num_visual_tokens, -1)
        return x
