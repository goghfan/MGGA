import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TwoWayTransformer

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

class GeometricPerceptionReconstructionMR(nn.Module):
    def __init__(self, input_dim=256, low_rank_dim=32, 
                 implicit_input_size=(64, 64), implicit_output_size=(10, 12, 12), 
                 hidden_dim=128, 
                 fusion_metric='original'):
        super(GeometricPerceptionReconstructionMR, self).__init__()
        
        self.fusion_metric = fusion_metric
        self.input_dim = input_dim
        self.low_rank_dim = low_rank_dim
        
        self.implicit_input_spatial_area = implicit_input_size[0] * implicit_input_size[1]
        self.implicit_output_spatial_volume = implicit_output_size[0] * implicit_output_size[1] * implicit_output_size[2]

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

        if self.fusion_metric == 'original':
            self.mlp_alpha = nn.Sequential(
                nn.Linear(2, 1),
                nn.Sigmoid()
            )
        elif self.fusion_metric == 'mahalanobis':
            self.mahalanobis_scale = nn.Parameter(torch.ones(1, low_rank_dim, 1, 1, 1))
            self.mlp_alpha = nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            )
        elif self.fusion_metric == 'mi':
            self.local_window = 5
            self.avg_pool = nn.AvgPool3d(kernel_size=self.local_window, stride=1, padding=self.local_window//2)
            self.mlp_alpha = nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            )

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
        assert X * Y == self.implicit_input_spatial_area

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
        B, C, D, H, W = f_proj.shape
        
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
            
        return alpha.permute(0, 4, 1, 2, 3)

    def forward(self, features_2d, pos_2d, sparse_code_2d, dense_code_2d,
                features_3d, slice_positions, original_shape):
        
        output_shape_final = features_3d.shape[2:]
        
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
        
        low_rank_2d = self.mlp_2d(fused_2d_features)

        scale_factors = (1.0, low_rank_2d.shape[2] / features_3d.shape[3], low_rank_2d.shape[3] / features_3d.shape[4])
        features_3d_upsampled = F.interpolate(features_3d, scale_factor=scale_factors, mode='trilinear', align_corners=True)
        low_rank_3d = self.mlp_3d(features_3d_upsampled)

        target_shape_3d = low_rank_3d.shape[2:] 
        projection_3d = self.encode_project_2d_to_3d(
            input_matrix=low_rank_2d,
            original_size=original_shape,
            compressed_size=target_shape_3d,
            positions=slice_positions
        )
        
        alpha = self.calculate_alpha(projection_3d, low_rank_3d)

        fused_result = alpha * projection_3d + (1 - alpha) * low_rank_3d
        
        result = self.mlp_up(fused_result)
        if result.shape[2:] != output_shape_final:
            result = F.interpolate(result, size=output_shape_final, mode='trilinear', align_corners=False)

        return result

if __name__ == "__main__":
    B, C, H, W = 2, 256, 64, 64
    D, H_in, W_in = 10, 12, 12
    features_2d = torch.randn(B, 512, H, W) 
    pos_2d = torch.randn(B, C, H, W)
    sparse = torch.randn(B, 5, C)
    dense = torch.randn(B, C, H, W)
    features_3d = torch.randn(B, C, D, H_in, W_in)
    slice_pos = torch.randn(B, 3)
    orig_shape = (160, 400, 400)

    print("--- Experiment 1: Original (Cosine + Gaussian) ---")
    model_orig = GeometricPerceptionReconstructionMR(input_dim=256, fusion_metric='original')
    out1 = model_orig(features_2d, pos_2d, sparse, dense, features_3d, slice_pos, orig_shape)
    print(f"Output shape: {out1.shape}")

    print("\n--- Experiment 2: Comparison (Mahalanobis) ---")
    model_maha = GeometricPerceptionReconstructionMR(input_dim=256, fusion_metric='mahalanobis')
    out2 = model_maha(features_2d, pos_2d, sparse, dense, features_3d, slice_pos, orig_shape)
    print(f"Output shape: {out2.shape}")

    print("\n--- Experiment 3: Comparison (Mutual Information) ---")
    model_mi = GeometricPerceptionReconstructionMR(input_dim=256, fusion_metric='mi')
    out3 = model_mi(features_2d, pos_2d, sparse, dense, features_3d, slice_pos, orig_shape)
    print(f"Output shape: {out3.shape}")
