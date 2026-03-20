import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitMLP(nn.Module):
    """将 2D 特征 + 相对位置编码映射到 3D 压缩空间的 MLP。"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class GeometricPerceptionReconstructionMR(nn.Module):
    """GPR：融合 MedSAM 2D 特征与文本 prompt，将其重建为 3D bottleneck 特征。"""

    def __init__(
        self,
        input_dim=256,
        input_3d_channels=768,
        output_3d_channels=768,
        low_rank_dim=32,
        feature_size=(64, 64),
        output_compressed_size=(5, 6, 7),
    ):
        super().__init__()

        self.input_dim = input_dim

        self.mlp_2d = nn.Sequential(
            nn.Conv2d(input_dim, low_rank_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(low_rank_dim, low_rank_dim, kernel_size=1),
        )
        self.mlp_3d = nn.Sequential(
            nn.Conv3d(input_3d_channels, low_rank_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(low_rank_dim, low_rank_dim, kernel_size=1),
        )
        self.mlp_alpha = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

        self.mlp_up = nn.Sequential(
            nn.Conv3d(low_rank_dim, low_rank_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(low_rank_dim * 2, input_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(input_dim, output_3d_channels, kernel_size=1),
        )

        self.down_sample = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1),
        )

        from segment_anything.modeling.transformer import TwoWayTransformer

        self.transformer = TwoWayTransformer(depth=1, embedding_dim=self.input_dim, mlp_dim=2048, num_heads=8)

        self.iou_token = nn.Embedding(1, self.input_dim)
        self.mask_tokens = nn.Embedding(1, self.input_dim)

        self.mlp_in_dim = feature_size[0] * feature_size[1] + 3
        self.mlp_out_dim = output_compressed_size[0] * output_compressed_size[1] * output_compressed_size[2]
        self.project_mlp = ImplicitMLP(self.mlp_in_dim, 128, self.mlp_out_dim)

    def encode_project_2d_to_3d(self, input_matrix, original_size, compressed_size, positions):
        B, C, X, Y = input_matrix.shape
        original_size_tensor = torch.tensor(original_size, dtype=torch.float32).to(input_matrix.device)
        relative_positions = positions / original_size_tensor  # (B, 3)

        input_flat = input_matrix.view(B, C, -1)  # (B, C, X*Y)
        relative_positions_expanded = relative_positions.unsqueeze(1).expand(-1, C, -1)  # (B, C, 3)

        mlp_input = torch.cat([input_flat, relative_positions_expanded], dim=-1)  # (B, C, X*Y+3)
        mlp_input = mlp_input.view(B * C, -1)
        mlp_output = self.project_mlp(mlp_input)

        output_matrix = mlp_output.view(B, C, *compressed_size)
        return output_matrix

    def forward(self, features_2d, pos_2d, sparse_code_2d, dense_code_2d, features_3d, slice_positions, original_shape):
        output_shape = features_3d.shape
        features_2d = self.down_sample(features_2d)

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)  # (2, input_dim)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_code_2d.size(0), -1, -1).to(sparse_code_2d.device)
        tokens = torch.cat((output_tokens, sparse_code_2d), dim=1)

        if features_2d.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(features_2d, tokens.shape[0], dim=0)
        else:
            src = features_2d

        src = src + dense_code_2d
        pos_src = torch.repeat_interleave(pos_2d, tokens.shape[0], dim=0)

        b, c, h, w = src.shape
        _, fused_2d_features = self.transformer(src, pos_src, tokens)
        fused_2d_features = fused_2d_features.transpose(1, 2).view(b, c, h, w)

        low_rank_2d = self.mlp_2d(fused_2d_features)

        target_feat_size = output_shape[2:]  # (l, h, w)
        rel_pos_embeddings = self.encode_project_2d_to_3d(
            input_matrix=low_rank_2d,
            original_size=original_shape,
            compressed_size=target_feat_size,
            positions=slice_positions,
        )

        target_hw = target_feat_size[1:]
        low_rank_2d_resized = F.interpolate(low_rank_2d, size=target_hw, mode="bilinear", align_corners=True)
        projection_3d = low_rank_2d_resized.unsqueeze(2) + rel_pos_embeddings
        low_rank_3d = self.mlp_3d(features_3d)

        cosine_similarity = F.cosine_similarity(projection_3d, low_rank_3d, dim=1)
        diff = projection_3d - low_rank_3d
        distances = torch.norm(diff, dim=1)
        gaussian_similarity = torch.exp(-distances / (2 * 1.0 ** 2))

        similarity_input = torch.stack([cosine_similarity, gaussian_similarity], dim=-1)  # (B, l, h, w, 2)
        alpha = self.mlp_alpha(similarity_input)  # (B, l, h, w, 1)
        alpha = alpha.permute(0, 4, 1, 2, 3)  # (B, 1, l, h, w)

        result = alpha * projection_3d + (1 - alpha) * low_rank_3d
        result = self.mlp_up(result)
        return result

