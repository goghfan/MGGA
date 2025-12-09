import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TwoWayTransformer
# ==============================================================================
# 1. 辅助模块：对比学习损失函数 (训练时必须加这个Loss)
# ==============================================================================

class PixelWiseContrastiveLoss(nn.Module):
    """
    [新增] 像素级对比损失 (InfoNCE Loss)
    用于 'contrastive' 模式。
    逻辑：在同一空间位置的 (2D_proj, 3D) 是正样本对，不同位置的是负样本对。
    """
    def __init__(self, temperature=0.07, num_samples=1024):
        super(PixelWiseContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_samples = num_samples # 为了节省显存，不计算全图，只随机采样部分点

    def forward(self, proj_2d, feat_3d):
        """
        proj_2d, feat_3d: (B, C, D, H, W)
        """
        B, C, D, H, W = proj_2d.shape
        
        # 1. 展平并重排: (B, D*H*W, C)
        proj_2d_flat = proj_2d.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        feat_3d_flat = feat_3d.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        
        # 2. 随机采样 N 个点 (避免计算量爆炸)
        num_pixels = D * H * W
        # 如果像素总数少于采样数，就取全部
        actual_samples = min(self.num_samples, num_pixels)
        
        # 生成随机索引
        indices = torch.randperm(num_pixels, device=proj_2d.device)[:actual_samples]
        
        # 采样: (B, N, C)
        q = proj_2d_flat[:, indices, :] 
        k = feat_3d_flat[:, indices, :]
        
        # 3. 归一化 (Cosine Similarity 前置步骤)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # 4. 计算 Logits (B, N, N)
        # 每一行 i 代表第 i 个 2D 点，它应该与第 i 个 3D 点(对角线)匹配
        # bmm: batch matrix multiplication
        logits = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        
        # 5. 生成标签 (对角线是正样本)
        labels = torch.arange(actual_samples, dtype=torch.long, device=proj_2d.device)
        labels = labels.unsqueeze(0).expand(B, -1) # (B, N)
        
        # 6. 计算 Cross Entropy Loss
        # Flatten for CrossEntropy: logits (B*N, N), labels (B*N)
        loss = F.cross_entropy(logits.reshape(-1, actual_samples), labels.reshape(-1))
        
        return loss

class ImplicitMLP(nn.Module):
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

# ==============================================================================
# 3. 主模型 (集成 Contrastive 逻辑)
# ==============================================================================

class GeometricPerceptionReconstructionMR(nn.Module):
    def __init__(self, input_dim=256, low_rank_dim=32, 
                 implicit_input_size=(64, 64), implicit_output_size=(10, 12, 12), 
                 hidden_dim=128, 
                 fusion_metric='original'):
        """
        fusion_metric: 'original', 'mahalanobis', 'mi', 'contrastive'
        """
        super(GeometricPerceptionReconstructionMR, self).__init__()
        
        self.fusion_metric = fusion_metric
        self.input_dim = input_dim
        self.low_rank_dim = low_rank_dim
        
        self.implicit_input_spatial_area = implicit_input_size[0] * implicit_input_size[1]
        self.implicit_output_spatial_volume = implicit_output_size[0] * implicit_output_size[1] * implicit_output_size[2]

        # Encoders
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

        self.implicit_mlp = ImplicitMLP(
            input_dim=self.implicit_input_spatial_area + 3,  
            hidden_dim=hidden_dim,
            output_dim=self.implicit_output_spatial_volume        
        )

        # --- Fusion Metrics 配置 ---
        if self.fusion_metric == 'original':
            self.mlp_alpha = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

        elif self.fusion_metric == 'mahalanobis':
            self.mahalanobis_scale = nn.Parameter(torch.ones(1, low_rank_dim, 1, 1, 1))
            self.mlp_alpha = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

        elif self.fusion_metric == 'mi':
            self.local_window = 5
            self.avg_pool = nn.AvgPool3d(kernel_size=self.local_window, stride=1, padding=self.local_window//2)
            self.mlp_alpha = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

        elif self.fusion_metric == 'contrastive':
            # [新增] Contrastive Projection Head
            # 将特征投影到一个专门用于比较相似度的度量空间
            self.proj_head = nn.Sequential(
                nn.Conv3d(low_rank_dim, low_rank_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(low_rank_dim, low_rank_dim, kernel_size=1)
            )
            # 在 Contrastive 模式下，相似度就是 Normalized Dot Product (Cosine)
            # 所以这里不需要复杂的 MLP，只需要一个可学习的缩放与偏置即可，或者简单的 MLP
            self.mlp_alpha = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

        # Rest of the network
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
        self.transformer = TwoWayTransformer(depth=1, embedding_dim=self.input_dim, mlp_dim=2048, num_heads=8)
        self.iou_token = nn.Embedding(1, self.input_dim)
        self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.input_dim)

    def encode_project_2d_to_3d(self, input_matrix, original_size, compressed_size, positions):
        B, C, X, Y = input_matrix.shape
        original_size_tensor = torch.tensor(original_size, dtype=torch.float32, device=input_matrix.device)
        if positions.dim() == 1:
             positions = positions.unsqueeze(0).expand(B, -1)
        relative_positions = positions / original_size_tensor

        input_flat = input_matrix.reshape(B * C, -1) 
        relative_positions_expanded = relative_positions.unsqueeze(1).expand(B, C, 3).reshape(B * C, 3)
        mlp_input = torch.cat([input_flat, relative_positions_expanded], dim=1)
        mlp_output = self.implicit_mlp(mlp_input) 

        output_matrix = mlp_output.view(B, C, *compressed_size)
        return output_matrix

    def calculate_alpha(self, f_proj, f_3d):
        """
        计算融合权重，并返回用于 Loss 计算的中间变量（仅在 contrastive 模式下）
        """
        B, C, D, H, W = f_proj.shape
        proj_feat = None
        target_feat = None

        if self.fusion_metric == 'original':
            cosine_sim = F.cosine_similarity(f_proj, f_3d, dim=1).unsqueeze(-1)
            diff = f_proj - f_3d
            distances = torch.norm(diff, p=2, dim=1).unsqueeze(-1)
            sigma = 1.0 
            gaussian_sim = torch.exp(-distances.pow(2) / (2 * sigma ** 2))
            sim_input = torch.cat([cosine_sim, gaussian_sim], dim=-1)
            alpha = self.mlp_alpha(sim_input)
            
        elif self.fusion_metric == 'mahalanobis':
            diff = f_proj - f_3d
            weighted_diff = diff * self.mahalanobis_scale 
            mahal_dist = torch.norm(weighted_diff, p=2, dim=1).unsqueeze(-1)
            alpha = self.mlp_alpha(mahal_dist)
            
        elif self.fusion_metric == 'mi':
            mu_x = self.avg_pool(f_proj)
            mu_y = self.avg_pool(f_3d)
            x_c = f_proj - mu_x
            y_c = f_3d - mu_y
            cov = self.avg_pool(x_c * y_c).sum(dim=1, keepdim=True)
            var_x = self.avg_pool(x_c ** 2).sum(dim=1, keepdim=True)
            var_y = self.avg_pool(y_c ** 2).sum(dim=1, keepdim=True)
            epsilon = 1e-5
            rho = cov / (torch.sqrt(var_x * var_y) + epsilon)
            rho = rho.permute(0, 2, 3, 4, 1)
            alpha = self.mlp_alpha(rho)
            
        elif self.fusion_metric == 'contrastive':
            # --- Contrastive Logic ---
            # 1. 投影到度量空间 (Metric Space)
            proj_feat = self.proj_head(f_proj) # z_2d
            target_feat = self.proj_head(f_3d) # z_3d
            
            # 2. 计算学习到的余弦相似度 (Learned Cosine Similarity)
            # 训练时，Loss 会强迫这里的正样本对趋近 1，负样本对趋近 -1
            sim = F.cosine_similarity(proj_feat, target_feat, dim=1).unsqueeze(-1) # (B, D, H, W, 1)
            
            # 3. 映射到 alpha [0, 1]
            alpha = self.mlp_alpha(sim)

        return alpha.permute(0, 4, 1, 2, 3), proj_feat, target_feat

    def forward(self, features_2d, pos_2d, sparse_code_2d, dense_code_2d,
                features_3d, slice_positions, original_shape):
        
        output_shape_final = features_3d.shape[2:]
        
        # ... Preprocessing ...
        features_2d_down = self.down_sample(features_2d)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_code_2d.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_code_2d), dim=1).to(features_2d.device)
        src = features_2d_down
        if src.shape[0] != tokens.shape[0]: src = src.repeat_interleave(tokens.shape[0], dim=0)
        src = src + dense_code_2d
        pos_src = torch.repeat_interleave(pos_2d, tokens.shape[0] // pos_2d.shape[0], dim=0)
        fused_2d_features, _ = self.transformer(src, pos_src, tokens)
        
        # ... Encoding ...
        low_rank_2d = self.mlp_2d(fused_2d_features)
        scale_factors = (1.0, low_rank_2d.shape[2] / features_3d.shape[3], low_rank_2d.shape[3] / features_3d.shape[4])
        features_3d_upsampled = F.interpolate(features_3d, scale_factor=scale_factors, mode='trilinear', align_corners=True)
        low_rank_3d = self.mlp_3d(features_3d_upsampled)

        # ... Projection ...
        target_shape_3d = low_rank_3d.shape[2:] 
        projection_3d = self.encode_project_2d_to_3d(
            input_matrix=low_rank_2d,
            original_size=original_shape,
            compressed_size=target_shape_3d,
            positions=slice_positions
        )
        
        # ... Fusion ...
        # 注意：这里返回了额外的 feat 用于计算 loss
        alpha, contrast_feat_2d, contrast_feat_3d = self.calculate_alpha(projection_3d, low_rank_3d)

        fused_result = alpha * projection_3d + (1 - alpha) * low_rank_3d
        
        result = self.mlp_up(fused_result)
        if result.shape[2:] != output_shape_final:
            result = F.interpolate(result, size=output_shape_final, mode='trilinear', align_corners=False)

        # 返回结果：为了训练方便，如果是 contrastive 模式，我们需要把用于计算 Loss 的特征也返回出去
        if self.fusion_metric == 'contrastive':
            return result, contrast_feat_2d, contrast_feat_3d
        else:
            return result

# ==========================================
# 训练循环示例 (非常重要！!)
# ==========================================
if __name__ == "__main__":
    # 模拟输入
    B, C, H, W = 2, 256, 64, 64
    features_2d = torch.randn(B, 512, H, W) 
    pos_2d = torch.randn(B, 256, H, W)
    sparse = torch.randn(B, 5, 256)
    dense = torch.randn(B, 256, H, W)
    features_3d = torch.randn(B, 256, 10, 12, 12)
    slice_pos = torch.randn(B, 3)
    orig_shape = (160, 400, 400)

    # 1. 实例化模型和 Contrastive Loss
    print("--- Contrastive Learning Mode ---")
    model_contrast = GeometricPerceptionReconstructionMR(input_dim=256, fusion_metric='contrastive')
    criterion_contrast = PixelWiseContrastiveLoss(temperature=0.07) # 初始化 Loss

    # 2. Forward Pass
    # 注意接收返回值：现在返回了 3 个变量
    output, z_2d, z_3d = model_contrast(features_2d, pos_2d, sparse, dense, features_3d, slice_pos, orig_shape)
    
    # 3. 计算额外的 Contrastive Loss
    # z_2d 和 z_3d 就是通过 Projection Head 映射后的特征
    loss_contrast = criterion_contrast(z_2d, z_3d)

    print(f"Main Output shape: {output.shape}")
    print(f"Contrastive Loss: {loss_contrast.item()}")
    print("注意: 训练时 Total_Loss = L_reg + L_sim + 0.1 * L_contrast")
