import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TwoWayTransformer

class ImplicitMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImplicitMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class GeometricPerceptionReconstructionMR(nn.Module):
    def __init__(self, input_dim=256, low_rank_dim=32):
        super(GeometricPerceptionReconstructionMR, self).__init__()
        self.mlp_2d = nn.Sequential(
            nn.Conv2d(input_dim, low_rank_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(low_rank_dim, low_rank_dim, kernel_size=1)
        )
        self.mlp_3d = nn.Sequential(
            nn.Conv3d(input_dim, low_rank_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(low_rank_dim, low_rank_dim, kernel_size=1)
        )
        self.mlp_alpha = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        self.mlp_up = nn.Sequential(
            nn.Conv3d(low_rank_dim, low_rank_dim*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(low_rank_dim*2, input_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(input_dim, input_dim, kernel_size=2, stride=2)
        )
        self.down_sample = nn.Sequential(
            nn.Conv2d(input_dim*2, input_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1),
        )
        self.input_dim = input_dim
        self.transformer = TwoWayTransformer(depth=1,embedding_dim=self.input_dim,mlp_dim=2048, num_heads=8)
        self.low_rank_dim = low_rank_dim
    def encode_project_2d_to_3d(self,input_matrix, original_size, compressed_size, positions, hidden_dim=128):
        B, C, X, Y = input_matrix.shape
        original_size_tensor = torch.tensor(original_size, dtype=torch.float32).to(input_matrix.device)
        relative_positions = (positions / original_size_tensor)
        input_flat = input_matrix.view(B, C, -1)
        relative_positions_expanded = relative_positions.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        mlp_input = torch.cat([input_flat, relative_positions_expanded], dim=-1)
        mlp_input = mlp_input.view(B * C, -1)
        input_dim = X * Y + 3
        output_dim = compressed_size[0] * compressed_size[1] * compressed_size[2]
        mlp = ImplicitMLP(input_dim, hidden_dim, output_dim).to(input_matrix.device)
        mlp_output = mlp(mlp_input)
        mlp_output = mlp_output.view(B, C, -1)
        output_matrix = mlp_output.view(B, C, *compressed_size)
        return output_matrix
    def forward(self, features_2d, pos_2d, sparse_code_2d, dense_code_2d,
                features_3d, slice_positions, original_shape):
        output_shape = features_3d.shape
        self.up_sample=nn.Upsample(scale_factor=(1,features_2d.shape[2] / features_3d.shape[3],features_2d.shape[3] / features_3d.shape[4]), mode='trilinear', align_corners=True)
        features_3d = self.up_sample(features_3d)
        features_2d = self.down_sample(features_2d)
        self.iou_token = nn.Embedding(1, self.input_dim)
        self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.input_dim)
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_code_2d.size(0), -1, -1
        ).to(sparse_code_2d.device)
        tokens = torch.cat((output_tokens, sparse_code_2d), dim=1)
        if features_2d.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(features_2d, tokens.shape[0], dim=0)
        else:
            src = features_2d
        device = dense_code_2d.device
        src = src.to(device)
        src = src + dense_code_2d
        pos_src = torch.repeat_interleave(pos_2d, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        _,fused_2d_features =self.transformer(src, pos_src, tokens)
        fused_2d_features = fused_2d_features.transpose(1, 2).view(b, c, h, w)
        low_rank_2d = self.mlp_2d(fused_2d_features)
        low_rank_3d = self.mlp_3d(features_3d)
        target_shape = features_3d.shape
        rel_pos_embeddings = self.encode_project_2d_to_3d(input_matrix=low_rank_2d,original_size=original_shape, compressed_size=target_shape[2:],positions=slice_positions,hidden_dim=self.low_rank_dim)
        projection_3d = low_rank_2d.unsqueeze(2) + rel_pos_embeddings
        cosine_similarity = F.cosine_similarity(projection_3d.unsqueeze(dim=0), low_rank_3d.unsqueeze(dim=0), dim=0)
        positions =projection_3d.unsqueeze(dim=0)
        diff = positions - low_rank_3d.unsqueeze(dim=0)
        distances = torch.norm(diff,dim=0)
        sigma = 1.0
        gaussian_similarity = torch.exp(-distances / (2 * sigma ** 2))
        similarity_input = torch.stack([cosine_similarity, gaussian_similarity], dim=-1)
        alpha = self.mlp_alpha(similarity_input)
        alpha = alpha.mean(dim=-1)
        result = alpha * projection_3d + (1 - alpha) * low_rank_3d
        result = self.mlp_up(result)
        result = F.interpolate(result, size=(output_shape[2],output_shape[3],output_shape[4]), mode='trilinear', align_corners=False)
        return result

if __name__ == "__main__":
    model = GeometricPerceptionReconstructionMR()
    features_2d = torch.randn(1, 512, 64, 64)
    pos_2d = torch.randn(1, 256, 64, 64)
    sparse_code_2d = torch.randn(1, 77, 256)
    dense_code_2d = torch.randn(1, 256, 64, 64)
    features_3d = torch.randn(1, 256, 10, 12, 12)
    slice_positions = torch.tensor([80,192,192])
    original_shape = (160,192,192)
    output = model(features_2d, pos_2d, sparse_code_2d, dense_code_2d, features_3d, slice_positions, original_shape)
    print(output.shape)
