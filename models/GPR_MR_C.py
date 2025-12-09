import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Helper Modules
# ==============================================================================

class TwoWayTransformer(nn.Module):
    """
    一个简化的双向特征融合模块 (Lightweight Feature Fusion).
    用于替代原本复杂的 Transformer 进行特征交互。
    """
    def __init__(self, depth, embedding_dim, mlp_dim, num_heads):
        super(TwoWayTransformer, self).__init__()
        self.depth = depth
        self.norm = nn.LayerNorm(embedding_dim)
        
        # 使用 1x1 卷积模拟 Channel-wise 的交互
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embedding_dim, mlp_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim // 2, embedding_dim, kernel_size=1)
        )

    def forward(self, image_embedding, image_pos_embedding, token_embedding):
        """
        Args:
            image_embedding: (B, C, H, W)
            image_pos_embedding: (B, C, H, W)
            token_embedding: (B, num_tokens, C)
        Returns:
            fused_image_features: (B, C, H, W)
            token_embedding: (B, num_tokens, C) (未修改，原样返回)
        """
        # 1. 简单的特征叠加
        x = image_embedding + image_pos_embedding
        
        # 2. 残差连接 + 融合处理
        residual = x
        x = self.fusion_conv(x)
        x = x + residual
        
        # 3. 返回更新后的图像特征 (模拟 Transformer 的输出)
        return x, token_embedding


class ImplicitMLP(nn.Module):
    """
    隐式神经表示 MLP
    将 (特征 + 位置) 映射到 (目标体积特征)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImplicitMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # 增加 Norm 提高训练稳定性
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


# ==============================================================================
# Main Model
# ==============================================================================

class GeometricPerceptionReconstructionMR(nn.Module):
    def __init__(self, input_dim=256, low_rank_dim=32, 
                 implicit_input_size=(64, 64), implicit_output_size=(10, 12, 12), hidden_dim=128):
        """
        Args:
            input_dim: 输入特征通道数
            low_rank_dim: 低秩映射维度
            implicit_input_size: 隐式MLP预期的输入2D空间大小 (H, W)
            implicit_output_size: 隐式MLP预期的输出3D空间大小 (X, Y, Z)
            hidden_dim: MLP 隐藏层维度
        """
        super(GeometricPerceptionReconstructionMR, self).__init__()
        
        self.input_dim = input_dim
        self.low_rank_dim = low_rank_dim
        
        # 记录尺寸用于校验
        self.implicit_input_spatial_area = implicit_input_size[0] * implicit_input_size[1]
        self.implicit_output_spatial_volume = implicit_output_size[0] * implicit_output_size[1] * implicit_output_size[2]

        # 1. Mapping layers (降维)
        self.mlp_2d = nn.Sequential(
            nn.Conv2d(input_dim, low_rank_dim, kernel_size=1),
            nn.BatchNorm2d(low_rank_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_rank_dim, low_rank_dim, kernel_size=1)
        )
        self.mlp_3d = nn.Sequential(
            nn.Conv3d(input_dim, low_rank_dim, kernel_size=1),
            nn.BatchNorm3d(low_rank_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(low_rank_dim, low_rank_dim, kernel_size=1)
        )

        # 2. Implicit MLP (核心投影模块)
        # Input: (H*W) 像素特征 + 3 (相对坐标)
        # Output: (X*Y*Z) 体素特征
        self.implicit_mlp = ImplicitMLP(
            input_dim=self.implicit_input_spatial_area + 3,  
            hidden_dim=hidden_dim,
            output_dim=self.implicit_output_spatial_volume       
        )

        # 3. Alpha Fusion Weights
        self.mlp_alpha = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
        # 4. Upsampling / Restoration
        self.mlp_up = nn.Sequential(
            nn.Conv3d(low_rank_dim, low_rank_dim*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(low_rank_dim*2, input_dim, kernel_size=1),
        )
        
        self.down_sample = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1),
        )

        # 5. Token & Transformer
        self.transformer = TwoWayTransformer(depth=1, embedding_dim=self.input_dim, mlp_dim=2048, num_heads=8)
        self.iou_token = nn.Embedding(1, self.input_dim)
        self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.input_dim)

    def encode_project_2d_to_3d(self, input_matrix, original_size, compressed_size, positions):
        """
        利用 MLP 将 2D 特征投影到 3D 空间
        """
        B, C, X, Y = input_matrix.shape
        
        # 安全检查：确保输入特征图大小与 MLP 初始化时一致
        assert X * Y == self.implicit_input_spatial_area, \
            f"Input feature size ({X}x{Y}={X*Y}) does not match ImplicitMLP input size ({self.implicit_input_spatial_area})"

        # 归一化相对位置
        original_size_tensor = torch.tensor(original_size, dtype=torch.float32, device=input_matrix.device)
        # 确保 positions 是 (B, 3)
        if positions.dim() == 1:
             positions = positions.unsqueeze(0).expand(B, -1)
        relative_positions = positions / original_size_tensor  # [B, 3]

        # 准备 MLP 输入
        # 1. 展平图像特征: [B, C, X, Y] -> [B*C, X*Y]
        input_flat = input_matrix.reshape(B * C, -1) 
        
        # 2. 扩展位置编码: [B, 3] -> [B, C, 3] -> [B*C, 3]
        relative_positions_expanded = relative_positions.unsqueeze(1).expand(B, C, 3).reshape(B * C, 3)
        
        # 3. 拼接: [B*C, X*Y + 3]
        mlp_input = torch.cat([input_flat, relative_positions_expanded], dim=1)
        
        # 执行投影
        mlp_output = self.implicit_mlp(mlp_input)  # [B*C, O*P*Q]

        # 恢复形状: [B, C, O, P, Q]
        output_matrix = mlp_output.view(B, C, *compressed_size)

        return output_matrix

    def forward(self, features_2d, pos_2d, sparse_code_2d, dense_code_2d,
                features_3d, slice_positions, original_shape):
        
        output_shape_final = features_3d.shape[2:] # (D, H, W)
        
        # ------------------------------------------------------------------
        # Step 0: Preprocessing & Transformer Fusion
        # ------------------------------------------------------------------
        # 降采样并融合
        features_2d_down = self.down_sample(features_2d) # (B, input_dim, H_s, W_s)
        
        # 构建 Tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_code_2d.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_code_2d), dim=1).to(features_2d.device)
        
        # 维度对齐
        src = features_2d_down
        if src.shape[0] != tokens.shape[0]:
            src = src.repeat_interleave(tokens.shape[0], dim=0)
        
        src = src + dense_code_2d
        pos_src = torch.repeat_interleave(pos_2d, tokens.shape[0] // pos_2d.shape[0], dim=0)

        # Transformer 交互
        fused_2d_features, _ = self.transformer(src, pos_src, tokens)
        
        # ------------------------------------------------------------------
        # Step 1: Low Rank Projection
        # ------------------------------------------------------------------
        low_rank_2d = self.mlp_2d(fused_2d_features) # (B, low_rank, H_s, W_s)

        # 3D 特征空间对齐：上采样 3D 特征以匹配 2D 特征的空间分辨率 (用于后续融合计算)
        # 注意：这里我们假设 MLP 输出的压缩尺寸(compressed_size)应该与这里的空间尺寸一致
        # 或者我们定义 target_shape_3d 为 low_rank_2d 映射过去的目标
        
        # 为了让逻辑通顺，我们假设 features_3d 进来时较小，我们将其上采样到一个中间态
        # 或者直接使用 MLP_3D 映射
        scale_factors = (
            1.0, 
            low_rank_2d.shape[2] / features_3d.shape[3], 
            low_rank_2d.shape[3] / features_3d.shape[4]
        )
        features_3d_upsampled = F.interpolate(features_3d, scale_factor=scale_factors, mode='trilinear', align_corners=True)
        low_rank_3d = self.mlp_3d(features_3d_upsampled) # (B, low_rank, X, H_s, W_s)

        # ------------------------------------------------------------------
        # Step 2: 2D-to-3D Implicit Projection
        # ------------------------------------------------------------------
        # 目标尺寸即为 low_rank_3d 的空间尺寸 (X, H_s, W_s)
        target_shape_3d = low_rank_3d.shape[2:] 
        
        projection_3d = self.encode_project_2d_to_3d(
            input_matrix=low_rank_2d,
            original_size=original_shape,
            compressed_size=target_shape_3d,
            positions=slice_positions
        )
        
        # ------------------------------------------------------------------
        # Step 3: Adaptive Fusion (Alpha Blending)
        # ------------------------------------------------------------------
        # Cosine Similarity
        cosine_sim = F.cosine_similarity(projection_3d, low_rank_3d, dim=1) # (B, X, H, W)
        
        # Gaussian Similarity (L2 distance based)
        diff = projection_3d - low_rank_3d
        distances = torch.norm(diff, p=2, dim=1) # (B, X, H, W)
        sigma = 1.0 
        gaussian_sim = torch.exp(-distances.pow(2) / (2 * sigma ** 2))

        # Alpha Calculation
        similarity_input = torch.stack([cosine_sim, gaussian_sim], dim=-1) # (B, X, H, W, 2)
        alpha = self.mlp_alpha(similarity_input) # (B, X, H, W, 1)
        alpha = alpha.permute(0, 4, 1, 2, 3)     # (B, 1, X, H, W)

        # Fusion
        fused_result = alpha * projection_3d + (1 - alpha) * low_rank_3d
        
        # ------------------------------------------------------------------
        # Step 4: Final Restoration
        # ------------------------------------------------------------------
        result = self.mlp_up(fused_result)
        
        # 恢复到原始输入的 3D 分辨率
        if result.shape[2:] != output_shape_final:
            result = F.interpolate(result, size=output_shape_final, mode='trilinear', align_corners=False)

        return result

if __name__ == "__main__":
    # ==========================================
    # Test Block
    # ==========================================
    print("Initializing Model...")
    
    # Configuration
    B = 2
    C_in = 512    # 输入 2D 特征通道
    C_model = 256 # 模型内部处理维度
    H, W = 64, 64 # 2D 特征图大小 (注意: input_dim*2 后经过 downsample 变为此大小)
    
    # 3D Dimensions
    D_in, H_in, W_in = 10, 12, 12 # 输入 3D 特征的空间大小
    
    # 这里的 implicit_output_size 必须匹配 low_rank_3d 的空间大小
    # 代码中 low_rank_3d 是 features_3d 上采样 H,W 后的结果
    # features_3d: (X, 12, 12) -> Upsample to (X, 64, 64)
    target_D, target_H, target_W = D_in, H, W 

    model = GeometricPerceptionReconstructionMR(
        input_dim=C_model, 
        low_rank_dim=32,
        implicit_input_size=(H, W),
        implicit_output_size=(target_D, target_H, target_W),
        hidden_dim=128
    )
    
    # Dummy Data
    features_2d = torch.randn(B, C_in, H, W)       # Downsample 期望输入 C_in (512)
    pos_2d = torch.randn(B, C_model, H, W)
    sparse_code_2d = torch.randn(B, 5, C_model)    # 5 tokens
    dense_code_2d = torch.randn(B, C_model, H, W)
    features_3d = torch.randn(B, C_model, D_in, H_in, W_in)
    
    slice_positions = torch.tensor([[80, 192, 192], [85, 200, 200]], dtype=torch.float32) # (B, 3)
    original_shape = (160, 400, 400)

    # Forward Pass
    try:
        output = model(features_2d, pos_2d, sparse_code_2d, dense_code_2d, features_3d, slice_positions, original_shape)
        print(f"\nSuccess!")
        print(f"Input 3D shape:  {features_3d.shape}")
        print(f"Output 3D shape: {output.shape}")
    except Exception as e:
        print(f"\nError occurred: {e}")
