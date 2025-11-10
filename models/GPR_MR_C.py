import torch
import torch.nn as nn
import torch.nn.functional as F
# from .transformer import TwoWayTransformer # 原始导入

# ==============================================================================
# 为了让代码可独立运行，此处补充一个简化的 TwoWayTransformer 实现
# 注意：这是一个占位符实现，实际应用中应使用您项目中的真实 Transformer 模块
# ==============================================================================
class TwoWayTransformer(nn.Module):
    def __init__(self, depth, embedding_dim, mlp_dim, num_heads):
        super(TwoWayTransformer, self).__init__()
        # 这是一个简化的实现，仅为了让代码跑通
        # 它仅通过一个简单的卷积层来模拟特征融合的过程
        self.embedding_dim = embedding_dim
        self.conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
        print("Note: Using a simplified placeholder for TwoWayTransformer.")

    def forward(self, image_embedding, image_pos_embedding, token_embedding):
        """
        简化版 forward:
        - image_embedding: (B, C, H, W)
        - image_pos_embedding: (B, C, H, W)
        - token_embedding: (B, num_tokens, C)
        """
        # 实际的 Transformer 会在此处进行复杂的交叉注意力计算
        # 此处我们仅模拟它输出了一个更新后的图像特征
        fused_features = image_embedding + image_pos_embedding
        fused_features = self.conv(fused_features)
        
        # 模拟返回更新后的图像嵌入和 token 嵌入
        # [B, C, H, W], [B, num_tokens, C]
        return fused_features, token_embedding

class ImplicitMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        构造隐式MLP网络 (此部分未修改)
        :param input_dim: 输入维度（展平后的特征维度 + 位置维度）
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出维度（目标3D体积的展平维度）
        """
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
    def __init__(self, input_dim=256, low_rank_dim=32, 
                 implicit_input_dim=4096, implicit_output_dim=1440, hidden_dim=128):
        """
        构造函数已修改：
        - 添加了 implicit_input_dim, implicit_output_dim, hidden_dim 用于预先定义 ImplicitMLP
        """
        super(GeometricPerceptionReconstructionMR, self).__init__()
        
        self.input_dim = input_dim
        self.low_rank_dim = low_rank_dim

        # Mapping layers to project features into low-rank space
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

        self.implicit_mlp = ImplicitMLP(
            input_dim=implicit_input_dim + 3,  # 展平的2D特征(X*Y) + 3D相对位置(3)
            hidden_dim=hidden_dim,
            output_dim=implicit_output_dim      # 目标3D体积(O*P*Q)
        )
        # ------------------

        # MLP for alpha weight calculation
        self.mlp_alpha = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
        # Upsampling MLP to restore feature dimensions
        self.mlp_up = nn.Sequential(
            nn.Conv3d(low_rank_dim, low_rank_dim*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(low_rank_dim*2, input_dim, kernel_size=1),
        )
        
        # Downsampling layer for input 2D features
        # 注意: 原始代码的 features_2d 输入通道为 512，这里我们假设输入为 input_dim*2=512
        self.down_sample = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1),
        )

        self.transformer = TwoWayTransformer(depth=1, embedding_dim=self.input_dim, mlp_dim=2048, num_heads=8)
        
        # Token embeddings
        self.iou_token = nn.Embedding(1, self.input_dim)
        self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.input_dim)

    def encode_project_2d_to_3d(self, input_matrix, original_size, compressed_size, positions):
        """
        将2D矩阵编码到3D空间 (此方法已修改)
        :param input_matrix: 输入2D低秩特征矩阵，形状为 [B, C, X, Y]
        :param original_size: 原始3D空间大小元组, e.g., (160, 192, 192)
        :param compressed_size: 压缩后3D空间大小元组, e.g., (10, 12, 12)
        :param positions: 位置张量，形状为 [B, 3]
        :return: 编码后的3D矩阵，形状为 [B, C, O, P, Q]
        """
        B, C, X, Y = input_matrix.shape

        original_size_tensor = torch.tensor(original_size, dtype=torch.float32, device=input_matrix.device)
        relative_positions = positions / original_size_tensor  # [B, 3]

        input_flat = input_matrix.view(B, C, -1)  # [B, C, X*Y]

        # --- [逻辑微调] ---
        # 调整拼接方式以匹配MLP的输入
        # 我们将每个batch中的C个通道都视为独立的样本进行处理
        input_flat_permuted = input_flat.permute(0, 2, 1).reshape(B * (X * Y), C) # [B*X*Y, C] -> 不对，MLP应该处理整个图像
        
        input_flat = input_flat.view(B, -1) # [B, C*X*Y]
        
        # 拼接 2D 特征与相对位置
        # 输入维度应为 C*X*Y + 3
        # 注意：这里的实现逻辑是将整个2D平面的所有特征(C*X*Y)与一个3D位置向量拼接
        # 这意味着ImplicitMLP的输入维度会非常大。
        # 我们遵循原始代码的逻辑，将C看作是low_rank_dim
        # MLP的输入维度应该是 (X*Y) + 3，处理 C 个通道
        
        # 恢复原始逻辑：对每个通道独立处理
        input_flat = input_matrix.view(B * C, -1) # [B*C, X*Y]
        # 广播相对位置到每个通道
        relative_positions_expanded = relative_positions.unsqueeze(1).expand(B, C, 3).reshape(B * C, 3) # [B*C, 3]
        
        mlp_input = torch.cat([input_flat, relative_positions_expanded], dim=1)  # [B*C, X*Y + 3]
        
        # --- [核心修改] ---
        # 不再重新创建MLP，而是使用在 __init__ 中定义的 self.implicit_mlp
        mlp_output = self.implicit_mlp(mlp_input)  # [B*C, O*P*Q]

        # 恢复到目标3D形状
        output_matrix = mlp_output.view(B, C, *compressed_size)  # [B, C, O, P, Q]

        return output_matrix

    def forward(self, features_2d, pos_2d, sparse_code_2d, dense_code_2d,
                features_3d, slice_positions, original_shape):
        """
        前向传播函数已修改
        """
        output_shape_final = features_3d.shape[2:]
        
        # Upsample 3D features to match 2D feature resolution for potential fusion
        # 注意：这里的上采样逻辑可能需要根据具体任务调整
        upsample_3d = nn.Upsample(
            scale_factor=(1, features_2d.shape[2] / features_3d.shape[3], features_2d.shape[3] / features_3d.shape[4]), 
            mode='trilinear', align_corners=True
        ).to(features_3d.device)
        features_3d_upsampled = upsample_3d(features_3d)

        # Step 1: Fuse 2D image features
        features_2d_down = self.down_sample(features_2d) # Shape: (B, 256, 64, 64)
        
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        ).unsqueeze(0).expand(sparse_code_2d.size(0), -1, -1)
        
        tokens = torch.cat((output_tokens, sparse_code_2d), dim=1).to(features_2d.device)
        
        # Expand per-image data in batch direction to be per-mask if needed
        src = features_2d_down
        if src.shape[0] != tokens.shape[0]:
            src = src.repeat_interleave(tokens.shape[0], dim=0)
        
        src = src + dense_code_2d
        pos_src = torch.repeat_interleave(pos_2d, tokens.shape[0], dim=0)

        # --- [核心修改] ---
        # 修正 Transformer 的输出用法
        # 第一个返回值是更新后的图像特征，第二个是更新后的 token
        fused_2d_features, _ = self.transformer(src, pos_src, tokens)
        
        # Project fused 2D features to low-rank space
        low_rank_2d = self.mlp_2d(fused_2d_features) # Shape: (B, low_rank_dim, H, W)

        # Map 3D features to low-rank space
        low_rank_3d = self.mlp_3d(features_3d_upsampled)  # Shape: (B, low_rank_dim, X, H, W)

        # Step 2: Compute relative position embeddings and project to 3D
        target_shape_3d = low_rank_3d.shape
        rel_pos_embeddings = self.encode_project_2d_to_3d(
            input_matrix=low_rank_2d,
            original_size=original_shape,
            compressed_size=target_shape_3d[2:],
            positions=slice_positions
        )
        
        # 此处unsqueeze(2)的广播可能需要与rel_pos_embeddings的生成方式匹配
        # 假设 projection_3d 是2D信息在3D空间的表示
        projection_3d = rel_pos_embeddings  

        # Step 3: Combine 3D projection with original 3D features via dynamic weighting
        # 计算相似度以决定权重 alpha
        cosine_similarity = F.cosine_similarity(projection_3d, low_rank_3d, dim=1) # [B, X, H, W]
        
        diff = projection_3d - low_rank_3d
        distances = torch.norm(diff, p=2, dim=1) # [B, X, H, W]
        sigma = 1.0 
        gaussian_similarity = torch.exp(-distances.pow(2) / (2 * sigma ** 2))

        # 准备 alpha 计算的输入
        similarity_input = torch.stack([cosine_similarity, gaussian_similarity], dim=-1) # [B, X, H, W, 2]
        
        alpha = self.mlp_alpha(similarity_input) # [B, X, H, W, 1]
        alpha = alpha.permute(0, 4, 1, 2, 3) # [B, 1, X, H, W]

        result = alpha * projection_3d + (1 - alpha) * low_rank_3d
        
        # Step 4: Upsample result to final feature dimension and size
        result = self.mlp_up(result)
        result = F.interpolate(result, size=output_shape_final, mode='trilinear', align_corners=False)

        return result

if __name__ == "__main__":
    # Example usage
    B, C_in, H, W = 1, 512, 64, 64
    C_model = 256
    X, Y, Z = 10, 12, 12
    low_rank = 32
    
    # --- [核心修改] ---
    # 计算 ImplicitMLP 所需的维度
    # 输入维度 = H * W
    # 输出维度 = X * Y * Z
    implicit_input = H * W
    implicit_output = X * Y * Z
    
    # 实例化模型时传入维度参数
    model = GeometricPerceptionReconstructionMR(
        input_dim=C_model, 
        low_rank_dim=low_rank,
        implicit_input_dim=implicit_input,
        implicit_output_dim=implicit_output
    )
    
    # 创建符合输入尺寸的随机数据
    features_2d = torch.randn(B, C_in, H, W)
    pos_2d = torch.randn(B, C_model, H, W)
    sparse_code_2d = torch.randn(B, 77, C_model) # 假设有77个 token
    dense_code_2d = torch.randn(B, C_model, H, W)
    features_3d = torch.randn(B, C_model, X, Y, Z)
    slice_positions = torch.tensor([[80, 192, 192]], dtype=torch.float32) # [B, 3]
    original_shape = (160, 192, 192)

    # 运行模型
    output = model(features_2d, pos_2d, sparse_code_2d, dense_code_2d, features_3d, slice_positions, original_shape)
    
    print("Input 3D shape:", features_3d.shape)
    print("Output 3D shape:", output.shape)
    
    # 验证输出形状是否与输入3D形状一致
    assert features_3d.shape == output.shape
    print("\nCode runs successfully! Output shape matches input 3D shape.")