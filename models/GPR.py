import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TwoWayTransformer 

class GeometricPerceptionReconstruction(nn.Module):
    def __init__(self, input_dim=256, low_rank_dim=32, original_size=(128, 512, 512), target_shape=(8, 32, 32)):
        """
        需要预先知道 original_size 和 target_shape 来初始化 MLP
        """
        super(GeometricPerceptionReconstruction, self).__init__()
        
        self.input_dim = input_dim
        self.low_rank_dim = low_rank_dim
        self.target_shape = target_shape
        self.original_size = original_size

        # 1. 初始化 Tokens (移到 init)
        self.iou_token = nn.Embedding(1, self.input_dim)
        self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.input_dim)

        # 2. 映射层
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

        # 3. 初始化 ImplicitMLP (移到 init)
        # 注意：这里假设输入特征图大小固定为 64x64 (根据你的 main 示例)
        # 如果特征图大小变化，这种 MLP 写法是不行的，需要改为 Conv 或者 Coordinate-based MLP
        feature_size_h, feature_size_w = 64, 64 
        mlp_in_dim = (feature_size_h * feature_size_w) + 3 
        mlp_out_dim = target_shape[0] * target_shape[1] * target_shape[2]
        
        self.implicit_projector = ImplicitMLP(mlp_in_dim, 128, mlp_out_dim)

        # 4. Alpha 计算
        self.mlp_alpha = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

        # 5. 上采样与融合
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

        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.transformer = TwoWayTransformer(depth=1, embedding_dim=self.input_dim, mlp_dim=2048, num_heads=8)

    def encode_project_2d_to_3d(self, input_matrix, positions):
        B, C, X, Y = input_matrix.shape

        # 归一化位置
        original_size_tensor = torch.tensor(self.original_size, dtype=torch.float32).to(input_matrix.device)
        
        # 确保 positions 维度正确 (B, 3)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).repeat(B, 1) # (B, 3)

        relative_positions = (positions / original_size_tensor) 

        # Flatten 输入矩阵
        input_flat = input_matrix.view(B, C, -1)  # [B, C, X*Y]

        # 广播位置
        relative_positions_expanded = relative_positions.unsqueeze(1).expand(B, C, -1) # [B, C, 3]

        # 拼接
        mlp_input = torch.cat([input_flat, relative_positions_expanded], dim=-1) # [B, C, X*Y+3]

        # 通过已经定义好的 MLP
        # view 为 (B*C, -1) 以批量处理通道
        mlp_input = mlp_input.view(B * C, -1)
        mlp_output = self.implicit_projector(mlp_input) 

        # 恢复形状 [B, C, D, H, W]
        output_matrix = mlp_output.view(B, C, *self.target_shape)
        
        return output_matrix

    def forward(self, features_2d, pos_2d, sparse_code_2d, dense_code_2d, 
                features_3d, slice_positions, original_shape=None):
        
        # Step 1: 预处理
        features_3d = self.up_sample(features_3d) # (B, C, D*2, H*2, W*2) -> 假设这里是上采样
        features_2d_down = self.down_sample(features_2d) # (B, C, H, W)

        # 准备 Token
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_code_2d.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_code_2d), dim=1)

        # Transformer Fusion
        src = features_2d_down + dense_code_2d
        # 注意：这里需要确保维度匹配，假设 transformer 接受这些输入
        pos_src = torch.repeat_interleave(pos_2d, tokens.shape[0]//pos_2d.shape[0], dim=0)
        
        # 调用 transformer
        _, fused_2d_features = self.transformer(src, pos_src, tokens)
        
        # Reshape transformer output back to 2D map
        b, c, h, w = src.shape
        # 注意: transformer输出通常是 (B, L, C)，这里假设它是 flatten 的 sequence，需要 view 回去
        # 根据你的原始代码逻辑调整
        fused_2d_features = fused_2d_features.view(b, c, h, w) 

        low_rank_2d = self.mlp_2d(fused_2d_features)
        low_rank_3d = self.mlp_3d(features_3d)

        # Step 2: 投影 2D -> 3D
        # 使用 self.encode_project_2d_to_3d 而不是在内部定义
        rel_pos_embeddings = self.encode_project_2d_to_3d(low_rank_2d, slice_positions)
        
        # 融合: 2D 特征 unsqueeze 后加上投影的位置编码
        # 注意：这里 low_rank_2d (B, C, H, W) 需要扩充维度匹配 rel_pos (B, C, D, H, W)
        # 如果 D 维度对应 slice，这里直接相加可能物理意义不明，但代码逻辑上是可行的
        projection_3d = low_rank_2d.unsqueeze(2) + rel_pos_embeddings

        # Step 3: 计算相似度与融合
        # 修正：通常是计算 projection_3d 和 low_rank_3d 之间的相似度，不应该做 (B, B) 的全对比，除非是对比学习
        # 这里改为 Element-wise 对比 (B, C, D, H, W)
        
        # Cosine Sim (沿 Channel 维度)
        cosine_sim = F.cosine_similarity(projection_3d, low_rank_3d, dim=1).unsqueeze(1) # (B, 1, D, H, W)
        
        # L2 Distance / Gaussian
        diff = projection_3d - low_rank_3d
        dist = torch.norm(diff, dim=1, keepdim=True) # (B, 1, D, H, W)
        sigma = 1.0
        gaussian_sim = torch.exp(-dist / (2 * sigma ** 2))

        # Stack 特征用于计算 alpha
        # 为了过 MLP (Linear(2,1))，我们需要把最后维度变成 2
        # 当前形状 (B, 1, D, H, W)，我们需要 permute
        sim_input = torch.cat([cosine_sim, gaussian_sim], dim=1) # (B, 2, D, H, W)
        sim_input = sim_input.permute(0, 2, 3, 4, 1) # (B, D, H, W, 2)
        
        alpha = self.mlp_alpha(sim_input) # (B, D, H, W, 1)
        alpha = alpha.permute(0, 4, 1, 2, 3) # (B, 1, D, H, W)

        # 最终融合
        result = alpha * projection_3d + (1 - alpha) * low_rank_3d
        result = self.mlp_up(result)

        return result

if __name__ == "__main__":
    # Example usage
    # 调整 features_2d 通道为 512 以匹配 input_dim*2 (如果 input_dim=256)
    model = GeometricPerceptionReconstruction(input_dim=256, target_shape=(8, 32, 32))
    
    features_2d = torch.randn(1, 512, 64, 64) 
    pos_2d = torch.randn(1, 256, 64, 64)
    sparse_code_2d = torch.randn(1, 77, 256)
    dense_code_2d = torch.randn(1, 256, 64, 64)
    features_3d = torch.randn(1, 256, 4, 16, 16) # 假设输入小一点，被 upsample
    slice_positions = torch.tensor([64.0, 512.0, 512.0]) # Float
    
    # 模拟 batch size = 1
    slice_positions = slice_positions.unsqueeze(0) 

    output = model(features_2d, pos_2d, sparse_code_2d, dense_code_2d, features_3d, slice_positions)
    print("Output shape:", output.shape)
