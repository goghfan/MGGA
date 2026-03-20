import torch
import torch.nn as nn


class MedSAMRegNet(nn.Module):
    """
    将 MedSAM（文本+2D 图像特征）与 ViT-V-Net（3D 配准）融合：
    用 GPR 把 MedSAM 条件特征重建为 ViT-V-Net 解码所需的 3D bottleneck token 特征。
    """

    def __init__(self, image_encoder, prompt_encoder, register_model, gpr, img_size):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.register_model = register_model
        self.GPR = gpr
        self.img_size = img_size

        vit_config = getattr(self.register_model, "config", None)
        if vit_config is None:
            raise ValueError("register_model must have attribute `config`.")

        patch_size = vit_config.patches["size"]
        down_factor = vit_config.down_factor
        self.l = img_size[0] // (2 ** down_factor) // patch_size[0]
        self.h = img_size[1] // (2 ** down_factor) // patch_size[1]
        self.w = img_size[2] // (2 ** down_factor) // patch_size[2]

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        if hasattr(self.prompt_encoder, "text_encoder_head"):
            for param in self.prompt_encoder.text_encoder_head.parameters():
                param.requires_grad = True

    def forward(self, fixed_3d, moving_3d, fixed_2d_slice, moving_2d_slice, tokens, slice_ids):
        with torch.no_grad():
            emb_fixed = self.image_encoder(fixed_2d_slice)
            emb_moving = self.image_encoder(moving_2d_slice)

        image_embedding_2d = torch.cat([emb_fixed, emb_moving], dim=1)
        sparse_emb, dense_emb = self.prompt_encoder(tokens=tokens)

        # ViTVNet 内部 forward 里 source = x[:,0:1,...]，因此这里 x 的第 0 通道必须是 moving
        x_in = torch.cat([moving_3d, fixed_3d], dim=1)
        source = x_in[:, 0:1, :, :, :]

        encoded_tokens, _, vit_features = self.register_model.transformer(x_in)  # (B, n_patch, hidden)
        b, n_patch, hidden = encoded_tokens.shape
        features_3d = encoded_tokens.permute(0, 2, 1).contiguous().view(b, hidden, self.l, self.h, self.w)

        original_shape = fixed_3d.shape[2:]  # (D, H, W)
        slice_positions_batch = torch.stack(
            [
                slice_ids.float(),
                torch.full_like(slice_ids, original_shape[1], dtype=torch.float32),
                torch.full_like(slice_ids, original_shape[2], dtype=torch.float32),
            ],
            dim=1,
        ).to(fixed_3d.device)

        fused_embedding = self.GPR(
            features_2d=image_embedding_2d,
            pos_2d=self.prompt_encoder.get_dense_pe(),
            sparse_code_2d=sparse_emb,
            dense_code_2d=dense_emb,
            features_3d=features_3d,
            slice_positions=slice_positions_batch,
            original_shape=original_shape,
        )  # (B, hidden, l, h, w)

        # DecoderCup expects (B, n_patch, hidden)
        fused_tokens = fused_embedding.flatten(2).transpose(1, 2).contiguous()
        x_dec = self.register_model.decoder(fused_tokens, features=vit_features)
        flow = self.register_model.reg_head(x_dec)  # (B, 3, D, H, W)

        warped_moving = self.register_model.spatial_trans(source, flow)
        return warped_moving, flow

