import torch
import torch.nn as nn
from transformers import CLIPTextModel

from segment_anything.modeling import PromptEncoder


class TextPromptEncoder(PromptEncoder):
    """Text prompt encoder: 用 CLIPTextModel 编码 tokens，并输出 sparse/dense prompt embedding。"""

    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans: int = 1,
        activation=nn.GELU,
        clip_model_name_or_path: str = "openai/clip-vit-base-patch16",
    ):
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)

        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name_or_path)
        self.text_encoder.requires_grad_(False)

        # CLIPTextModel hidden size is 512 for this CLIP variant
        self.text_encoder_head = nn.Linear(512, embed_dim)

    def forward(self, tokens, masks=None, boxes=None, points=None):
        bs = tokens.shape[0]
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=tokens.device)

        if tokens is not None:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(tokens)[0]  # (B, seq_len, 512)
            text_embeddings = self.text_encoder_head(encoder_hidden_states)  # (B, seq_len, embed_dim)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
            )

        return sparse_embeddings, dense_embeddings

