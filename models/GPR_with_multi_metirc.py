import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Helper Modules
# ==============================================================================

class TwoWayTransformer(nn.Module):
    """
    一个简化的双向特征融合模块 (Lightweight Feature Fusion).
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
        # 1. 简单的特征叠加
        x = image_embedding + image_pos_embedding
        
        # 2. 残差连接 + 融合处理
        residual = x
        x = self.fusion_conv(x)
        x = x + residual
        
        # 3. 返回更新后的图像特征
        return x, token_embedding


class ImplicitMLP(nn.Module):
    """
    隐式神经表示 MLP
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImplicitMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class FusionMetricCalculator(nn.Module):
    """
    [新增模块] 专门用于计算多种几何与统计相似度
    包含: Cosine, Gaussian, Mahalanobis, Mutual Information (Local)
    """
    def __init__(self, channel_dim):
        super(FusionMetricCalculator, self).__init__()
        # Mahalanobis: 可学习的对角逆协方差矩阵权重 (Learnable Inverse Covariance)
        self.mahalanobis_scale = nn.Parameter(torch.ones(1, channel_dim, 1, 1, 1))
        
        # Mutual Information: 用于计算局部统计量的平均池化核
        self.local_window_size = 5
        self.avg_pool = nn.AvgPool3d(kernel_size=self.local_window_size, stride=1, padding=self.local_window_size//2)

    def compute_local_mi(self, x, y):
        """
        基于局部高斯假设的互信息近似 (Local Mutual Information via Local Correlation)
        MI(X,Y) = H(X) + H(Y) - H(X,Y)
        对于高斯分布，MI 与 -0.5*log(1-rho^2) 相关。
        这里我们计算局部相关系数 rho，并将其映射作为 MI 特征。
        """
        # 1. 计算局部均值
        mu_x = self.avg_pool(x)
        mu_y = self.avg_pool(y)
        
        # 2. 计算中心化变量
        x_centered = x - mu_x
        y_centered = y - mu_y
        
        # 3. 计算局部方差和协方差
        var_x = self.avg_pool(x_centered ** 2)
        var_y = self.avg_pool(y_centered ** 2)
        cov_xy = self.avg_pool(x_centered * y_centered)
        
        # 4. 计算局部相关系数 (rho)
        # 添加 epsilon 防止除零
        epsilon = 1e-5
        rho = cov_xy / (torch.sqrt(var_x * var_y) + epsilon)
        
        # 5. 映射为 MI 近似值 (取绝对值表示相关强度，或者平方)
        # MI 理论上是正值，相关性越强 MI 越大
        mi_proxy = rho ** 2 
        
        # 由于我们希望输出是一个 feature map (B, 1, D, H, W)，我们在 Channel 维度取平均
        return torch.mean(mi_proxy, dim=1, keepdim=True)

    def forward(self, f1, f2):
        """
        f1, f2: (B, C, D, H, W)
        Returns: concatenated metrics (B, 4, D, H, W)
        """
        # 1. Cosine Similarity (Feature Direction)
        cosine_sim = F.cosine_similarity(f1, f2, dim=1).unsqueeze(1) # (B, 1, D, H, W)
        
        # 2. Gaussian Similarity (Euclidean Distance based)
        diff = f1 - f2
        dist_sq = torch.sum(diff ** 2, dim=1, keepdim=True)
        gaussian_sim = torch.exp(-dist_sq / 2.0) # sigma=1.0 assumption
        
        # 3. Mahalanobis Distance (Learnable Weighted Distance)
        # Weighting the difference by learned scale (approx inverse covariance)
        weighted_diff = diff * self.mahalanobis_scale
        mahal_dist_sq = torch.sum(weighted_diff ** 2, dim=1, keepdim=True)
        mahal_sim = torch.exp(-mahal_dist_sq / 2.0) # 映射到 [0,1] 相似度区间
        
        # 4. Local Mutual Information (Statistical Dependency)
        mi_sim = self.compute_local_mi(f1, f2)
        
        # Concatenate all 4 metrics
        # Output shape: (B, 4, D, H, W)
        return torch.cat([cosine_sim, gaussian_sim, mahal_sim, mi_sim], dim=1)


# ==============================================================================
# Main Model
# ==============================================================================

class GeometricPerceptionReconstructionMR(nn.Module):
    def __init__(self, input_dim=256, low_rank_dim=32, 
                 implicit_input_size=(64, 64), implicit_output_size=(10, 12, 12), hidden_dim=128):
        super(GeometricPerceptionReconstructionMR, self).__init__()
        
        self.input_dim = input_dim
        self.low_rank_dim = low_rank_dim
        
        self.implicit_input_spatial_area = implicit_input_size[0] * implicit_input_size[1]
        self.implicit_output_spatial_volume = implicit_output_size[0] * implicit_output_size[1] * implicit_output_size[2]

        # 1. Mapping layers
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

        # 2. Implicit MLP
        self.implicit_mlp = ImplicitMLP(
            input_dim=self.implicit_input_spatial_area + 3,  
            hidden_dim=hidden_dim,
            output_dim=self.implicit_output_spatial_volume        
        )

        # 3. [修改] Metric Calculator (包含 MI 和 Mahalanobis)
        self.metric_calculator = FusionMetricCalculator(channel_dim=low_rank_dim)

        # 4. [修改] Alpha Fusion Weights
        # 输入维度变为 4 (Cos, Gauss, Mahal, MI)
        self.mlp_alpha = nn.Sequential(
            nn.Linear(4, 16), # 先升维做特征交互
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 5. Upsampling / Restoration
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

        # 6. Token & Transformer
        self.transformer = TwoWayTransformer(depth=1, embedding_dim=self.input_dim, mlp_dim=2048, num_heads=8)
        self.iou_token = nn.Embedding(1, self.input_dim)
        self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.input_dim)

    def encode_project_2d_to_3d(self, input_matrix, original_size, compressed_size, positions):
        B, C, X, Y = input_matrix.shape
        assert X * Y == self.implicit_input_spatial_area, \
            f"Input feature size ({X}x{Y}) mismatch."

        original_size_tensor = torch.tensor(original_size, dtype=torch.float32, device=input_matrix.device)
        if positions.dim() == 1:
             positions = positions.unsqueeze(0).expand(B, -1)
        relative_positions = positions / original_size_tensor  # [B, 3]

        input_flat = input_matrix.reshape(B * C, -1) 
        relative_positions_expanded = relative_positions.unsqueeze(1).expand(B, C, 3).reshape(B * C, 3)
        mlp_input = torch.cat([input_flat, relative_positions_expanded], dim=1)
        mlp_output = self.implicit_mlp(mlp_input) 

        output_matrix = mlp_output.view(B, C, *compressed_size)
        return output_matrix

    def forward(self, features_2d, pos_2d, sparse_code_2d, dense_code_2d,
                features_3d, slice_positions, original_shape):
        
        output_shape_final = features_3d.shape[2:] 
        
        # Step 0: Preprocessing
        features_2d_down = self.down_sample(features_2d)
        
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_code_2d.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_code_2d), dim=1).to(features_2d.device)
        
        src = features_2d_down
        if src.shape[0] != tokens.shape[0]:
            src = src.repeat_interleave(tokens.shape[0], dim=0)
        
        src = src + dense_code_2d
        pos_src = torch.repeat_interleave(pos_2d, tokens.shape[0] // pos_2d.shape[0], dim=0)

        fused_2d_features, _ = self.transformer(src, pos_src, tokens)
        
        # Step 1: Low Rank Projection
        low_rank_2d = self.mlp_2d(fused_2d_features)

        # 3D Feature Upsampling to match intermediate fusion resolution
        scale_factors = (
            1.0, 
            low_rank_2d.shape[2] / features_3d.shape[3], 
            low_rank_2d.shape[3] / features_3d.shape[4]
        )
        features_3d_upsampled = F.interpolate(features_3d, scale_factor=scale_factors, mode='trilinear', align_corners=True)
        low_rank_3d = self.mlp_3d(features_3d_upsampled)

        # Step 2: 2D-to-3D Implicit Projection
        target_shape_3d = low_rank_3d.shape[2:] 
        projection_3d = self.encode_project_2d_to_3d(
            input_matrix=low_rank_2d,
            original_size=original_shape,
            compressed_size=target_shape_3d,
            positions=slice_positions
        )
        
        # ------------------------------------------------------------------
        # Step 3: Adaptive Fusion with Enhanced Metrics
        # ------------------------------------------------------------------
        # [修改] 调用 metric calculator 计算 4 种相似度
        # similarity_maps shape: (B, 4, X, H, W)
        similarity_maps = self.metric_calculator(projection_3d, low_rank_3d)

        # Permute for MLP input: (B, X, H, W, 4)
        similarity_input = similarity_maps.permute(0, 2, 3, 4, 1)
        
        # Calculate Alpha
        alpha = self.mlp_alpha(similarity_input) # (B, X, H, W, 1)
        alpha = alpha.permute(0, 4, 1, 2, 3)     # (B, 1, X, H, W)

        # Fusion
        fused_result = alpha * projection_3d + (1 - alpha) * low_rank_3d
        
        # Step 4: Final Restoration
        result = self.mlp_up(fused_result)
        
        if result.shape[2:] != output_shape_final:
            result = F.interpolate(result, size=output_shape_final, mode='trilinear', align_corners=False)

        return result

if __name__ == "__main__":
    print("Initializing Model...")
    B = 2
    C_in = 512    
    C_model = 256 
    H, W = 64, 64 
    D_in, H_in, W_in = 10, 12, 12 
    target_D, target_H, target_W = D_in, H, W 

    model = GeometricPerceptionReconstructionMR(
        input_dim=C_model, 
        low_rank_dim=32,
        implicit_input_size=(H, W),
        implicit_output_size=(target_D, target_H, target_W),
        hidden_dim=128
    )
    
    features_2d = torch.randn(B, C_in, H, W)
    pos_2d = torch.randn(B, C_model, H, W)
    sparse_code_2d = torch.randn(B, 5, C_model)
    dense_code_2d = torch.randn(B, C_model, H, W)
    features_3d = torch.randn(B, C_model, D_in, H_in, W_in)
    slice_positions = torch.tensor([[80, 192, 192], [85, 200, 200]], dtype=torch.float32)
    original_shape = (160, 400, 400)

    try:
        output = model(features_2d, pos_2d, sparse_code_2d, dense_code_2d, features_3d, slice_positions, original_shape)
        print(f"\nSuccess!")
        print(f"Metrics Integrated: Cosine, Gaussian, Mahalanobis, Mutual Information")
        print(f"Output 3D shape: {output.shape}")
    except Exception as e:
        print(f"\nError occurred: {e}")
