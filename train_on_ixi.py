import argparse
import os

import torch
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from segment_anything import sam_model_registry
from segment_anything.modeling import NCC_vxm, Grad3d

from models.ViTVNet import CONFIGS as CONFIGS_ViT
from models.ViTVNet import ViTVNet

from ixi_dataset import IXIDataset
from medsam_reg_net import MedSAMRegNet
from text_prompt_encoder import TextPromptEncoder
from gpr import GeometricPerceptionReconstructionMR
from registration_utils import SpatialTransformer, dice_score, log_images


def parse_tuple_3_int(s: str):
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected format like '160,192,224'.")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Glob pattern for moving volumes, e.g. ./Train/*.pkl")
    parser.add_argument("--atlas_path", type=str, required=True, help="Path to atlas.pkl containing (atlas_vol, atlas_seg)")

    parser.add_argument("--medsam_ckpt", type=str, default="work_dir/medsam_vit_b.pth")
    parser.add_argument("--sam_type", type=str, default="vit_b", help="sam_model_registry key, e.g. vit_b")
    parser.add_argument("--work_dir", type=str, default="experiments/medsam_reg")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 1.0, 0.5], help="Weights: w_ncc w_reg w_dice")

    parser.add_argument("--img_size", type=str, default="160,192,224", help="3D volume size: D,H,W")
    parser.add_argument("--medsam_size", type=int, default=1024, help="MedSAM 2D resize size")

    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch16", help="CLIP model name or local path")
    args = parser.parse_args()

    img_size = parse_tuple_3_int(args.img_size)

    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.work_dir, "logs"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dataset = IXIDataset(
        data_path=args.data_dir,
        atlas_path=args.atlas_path,
        img_size=img_size,
        medsam_size=args.medsam_size,
        clip_model_name_or_path=args.clip_model,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Models
    medsam_model = sam_model_registry[args.sam_type](checkpoint=args.medsam_ckpt)
    prompt_encoder = TextPromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=1,
        clip_model_name_or_path=args.clip_model,
    )

    config_vit = CONFIGS_ViT["ViT-V-Net"]
    vit_model = ViTVNet(config_vit, img_size=img_size)

    bottleneck_channels = config_vit.hidden_size
    gpr = GeometricPerceptionReconstructionMR(
        input_dim=256,
        input_3d_channels=bottleneck_channels,
        output_3d_channels=bottleneck_channels,
        feature_size=(64, 64),
        output_compressed_size=(5, 6, 7),
    )

    model = MedSAMRegNet(
        image_encoder=deepcopy(medsam_model.image_encoder),
        prompt_encoder=prompt_encoder,
        register_model=vit_model,
        gpr=gpr,
        img_size=img_size,
    ).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    sim_loss_fn = NCC_vxm()
    grad_loss_fn = Grad3d(penalty="l2")
    stn_label = SpatialTransformer(img_size, mode="nearest").to(device)

    w_ncc, w_reg, w_dice = args.weights
    global_step = 0
    best_loss = float("inf")

    print(f"Training started on {device}...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        pbar = tqdm(dataloader)

        for step, batch in enumerate(pbar):
            fixed = batch["fixed_image"].to(device)
            moving = batch["moving_image"].to(device)
            f_slice = batch["fixed_slice"].to(device)
            m_slice = batch["moving_slice"].to(device)
            tokens = batch["tokens"].to(device)
            slice_ids = batch["slice_id"].to(device)

            moving_seg = batch["moving_label"].to(device)
            fixed_seg = batch["fixed_label"].to(device)
            target_label_id = batch["text_label_id"].to(device)

            warped_moving, flow = model(fixed, moving, f_slice, m_slice, tokens, slice_ids)

            loss_ncc = sim_loss_fn(fixed, warped_moving)
            loss_reg = grad_loss_fn(flow)

            warped_seg = stn_label(moving_seg.float(), flow).long()
            bs = fixed.shape[0]

            loss_dice_batch = 0.0
            for i in range(bs):
                tid = target_label_id[i].item()
                pred_mask = (warped_seg[i] == tid).float()
                gt_mask = (fixed_seg[i] == tid).float()
                loss_dice_batch += (1.0 - dice_score(pred_mask, gt_mask))

            loss_dice = loss_dice_batch / bs
            total_loss = w_ncc * loss_ncc + w_reg * loss_reg + w_dice * loss_dice

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss.append(total_loss.item())
            writer.add_scalar("Train/Loss_Total", total_loss.item(), global_step)
            writer.add_scalar("Train/Loss_NCC", loss_ncc.item(), global_step)
            writer.add_scalar("Train/Loss_Reg", loss_reg.item(), global_step)
            writer.add_scalar("Train/Loss_Dice", loss_dice.item(), global_step)

            pbar.set_description(f"Ep {epoch} | Loss: {total_loss.item():.4f} | Dice: {1 - loss_dice:.3f}")
            global_step += 1

            if global_step % 50 == 0:
                log_images(writer, fixed, moving, warped_moving, fixed_seg, warped_seg, global_step)

        avg_loss = float(sum(epoch_loss) / max(1, len(epoch_loss)))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.work_dir, "best_model.pth"))
            print(f"New best model saved at epoch {epoch} with loss {best_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(args.work_dir, "latest_model.pth"))

    writer.close()


if __name__ == "__main__":
    main()

import os
import glob
import random
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from tqdm import tqdm
import argparse
from transformers import CLIPTokenizer, CLIPTextModel
from skimage import transform as skimage_transform
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import ml_collections

# --- 引入 Segment Anything 基础类 ---
from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder # 基类
from segment_anything.modeling import NCC_vxm, Grad3d

# --- 引入 ViT-V-Net 注册网络 ---
# 允许在两种目录结构下导入：
# 1) 你最终整理到 MIMR 后：`MIMR/models/*.py`
# 2) 当前工程中：`TransMorph_Transformer_for_Medical_Image_Registration-main/Baseline_Transformers/models/*.py`
try:
    from models.ViTVNet import CONFIGS as CONFIGS_ViT
    from models.ViTVNet import ViTVNet
except ModuleNotFoundError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vit_baseline_root = os.path.join(
        repo_root,
        "TransMorph_Transformer_for_Medical_Image_Registration-main",
        "Baseline_Transformers",
    )
    if vit_baseline_root not in sys.path:
        sys.path.append(vit_baseline_root)
    from models.ViTVNet import CONFIGS as CONFIGS_ViT
    from models.ViTVNet import ViTVNet

# ==========================================
# 1. 配置部分 (关键：强制关闭 ConvSkip 以匹配通道)
# ==========================================
def get_transmorph_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = False  # <--- 关键修复：关闭 ConvSkip，避免通道不匹配
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96       # 保持 96 (Standard)
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 16)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

# ==========================================
# 2. 基础组件 (DecoderBlock, TextEncoder)
# ==========================================
class TextPromptEncoder(PromptEncoder):
    """
    修复后的 PromptEncoder，增加了 CLIP 处理逻辑，支持输入 tokens
    """
    def __init__(self, embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=1, activation=nn.GELU):
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)
        try:
            # 请确保此路径存在，或者改为 "openai/clip-vit-base-patch16"
            clip_path = "/home/suzixian/nas/VLM_space/registration/MedSAM-main/openai_clip-vit-base-patch16"
            self.text_encoder = CLIPTextModel.from_pretrained(clip_path)
        except Exception as e:
            print(f"Warning: Local CLIP not found, downloading from HuggingFace... {e}")
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
            
        self.text_encoder.requires_grad_(False)
        self.text_encoder_head = nn.Linear(512, embed_dim)

    def forward(self, tokens, masks=None, boxes=None, points=None):
        bs = tokens.shape[0]
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=tokens.device)
        
        if tokens is not None:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(tokens)[0]
            text_embeddings = self.text_encoder_head(encoder_hidden_states)
            # 这里你可以选择 pooling 策略，目前保留序列作为 sparse prompts
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)
        
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return sparse_embeddings, dense_embeddings

class Conv3dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, use_batchnorm=True):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels) if use_batchnorm else nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*layers)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv3dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv3dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight.data.normal_(0, 1e-5)
        conv3d.bias.data.zero_()
        super().__init__(conv3d)

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0).type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

# ==========================================
# 3. TransMorph Decoder (本地修复版)
# ==========================================
class TransMorphDecoder(nn.Module):
    def __init__(self, config):
        super(TransMorphDecoder, self).__init__()
        self.embed_dim = config.embed_dim
        self.if_transskip = config.if_transskip
        self.if_convskip = config.if_convskip
        
        # 严格计算通道数
        self.up0 = DecoderBlock(self.embed_dim*8, self.embed_dim*4, skip_channels=self.embed_dim*4 if self.if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(self.embed_dim*4, self.embed_dim*2, skip_channels=self.embed_dim*2 if self.if_transskip else 0, use_batchnorm=False)
        self.up2 = DecoderBlock(self.embed_dim*2, self.embed_dim, skip_channels=self.embed_dim if self.if_transskip else 0, use_batchnorm=False)
        self.up3 = DecoderBlock(self.embed_dim, self.embed_dim//2, skip_channels=self.embed_dim//2 if self.if_convskip else 0, use_batchnorm=False)
        self.up4 = DecoderBlock(self.embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if self.if_convskip else 0, use_batchnorm=False)
        
        self.reg_head = RegistrationHead(config.reg_head_chan, 3, kernel_size=3)

    def forward(self, features):
        x = features[-1] # Bottleneck (8C)
        f1 = features[0] # C
        f2 = features[1] # 2C
        f3 = features[2] # 4C
        
        x = self.up0(x, f3)
        x = self.up1(x, f2)
        x = self.up2(x, f1)
        x = self.up3(x, None) 
        x = self.up4(x, None)
        
        flow = self.reg_head(x)
        return flow

# ==========================================
# 4. GPR 模块 (768 -> 768)
# ==========================================
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
    def forward(self, x): return self.mlp(x)

class GeometricPerceptionReconstructionMR(nn.Module):
    def __init__(self, input_dim=256, input_3d_channels=768, output_3d_channels=768, 
                 low_rank_dim=32, feature_size=(64, 64), output_compressed_size=(5, 6, 7)):
        super(GeometricPerceptionReconstructionMR, self).__init__()
        
        self.mlp_2d = nn.Sequential(
            nn.Conv2d(input_dim, low_rank_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(low_rank_dim, low_rank_dim, kernel_size=1)
        )
        self.mlp_3d = nn.Sequential(
            nn.Conv3d(input_3d_channels, low_rank_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(low_rank_dim, low_rank_dim, kernel_size=1)
        )
        self.mlp_alpha = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        
        # Output aligned to 768 to match TransMorph Bottleneck
        self.mlp_up = nn.Sequential(
            nn.Conv3d(low_rank_dim, low_rank_dim*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(low_rank_dim*2, input_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(input_dim, output_3d_channels, kernel_size=1) 
        )
        
        self.down_sample = nn.Sequential(
            nn.Conv2d(input_dim*2, input_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1),
        )

        self.input_dim = input_dim
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
        relative_positions = (positions / original_size_tensor) 
        input_flat = input_matrix.view(B, C, -1) 
        relative_positions_expanded = relative_positions.unsqueeze(1).expand(-1, C, -1)
        mlp_input = torch.cat([input_flat, relative_positions_expanded], dim=-1)
        mlp_input = mlp_input.view(B * C, -1)
        mlp_output = self.project_mlp(mlp_input)
        output_matrix = mlp_output.view(B, C, *compressed_size)
        return output_matrix

    def forward(self, features_2d, pos_2d, sparse_code_2d, dense_code_2d, features_3d, slice_positions, original_shape):
        output_shape = features_3d.shape
        features_2d = self.down_sample(features_2d)
        
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
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
        
        target_feat_size = output_shape[2:] 
        rel_pos_embeddings = self.encode_project_2d_to_3d(
            input_matrix=low_rank_2d,
            original_size=original_shape, 
            compressed_size=target_feat_size,
            positions=slice_positions
        )
        
        target_hw = target_feat_size[1:] 
        low_rank_2d_resized = F.interpolate(low_rank_2d, size=target_hw, mode='bilinear', align_corners=True)
        projection_3d = low_rank_2d_resized.unsqueeze(2) + rel_pos_embeddings
        low_rank_3d = self.mlp_3d(features_3d)
        
        cosine_similarity = F.cosine_similarity(projection_3d, low_rank_3d, dim=1)
        diff = projection_3d - low_rank_3d
        distances = torch.norm(diff, dim=1)
        gaussian_similarity = torch.exp(-distances / (2 * 1.0 ** 2))
        similarity_input = torch.stack([cosine_similarity, gaussian_similarity], dim=-1)
        alpha = self.mlp_alpha(similarity_input) 
        alpha = alpha.permute(0, 4, 1, 2, 3) 
        
        result = alpha * projection_3d + (1 - alpha) * low_rank_3d
        result = self.mlp_up(result)
        
        return result

# ==========================================
# 5. 主网络 Wrapper
# ==========================================
class MedSAMRegNet(nn.Module):
    def __init__(self, image_encoder, prompt_encoder, register_model, gpr, img_size):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.register_model = register_model
        self.GPR = gpr
        self.img_size = img_size
        self.vit_config = getattr(self.register_model, "config", None)
        if self.vit_config is None:
            raise ValueError("ViTVNet register_model must have attribute `config`.")

        # decoder 里会把 hidden tokens reshape 成 (B, hidden, l, h, w)
        patch_size = self.vit_config.patches["size"]
        down_factor = self.vit_config.down_factor
        self.l = img_size[0] // (2 ** down_factor) // patch_size[0]
        self.h = img_size[1] // (2 ** down_factor) // patch_size[1]
        self.w = img_size[2] // (2 ** down_factor) // patch_size[2]
        
        for param in self.image_encoder.parameters(): param.requires_grad = False
        for param in self.prompt_encoder.parameters(): param.requires_grad = False
        if hasattr(self.prompt_encoder, 'text_encoder_head'):
            for param in self.prompt_encoder.text_encoder_head.parameters(): param.requires_grad = True

    def forward(self, fixed_3d, moving_3d, fixed_2d_slice, moving_2d_slice, tokens, slice_ids):
        # 1. MedSAM Features
        with torch.no_grad():
            emb_fixed = self.image_encoder(fixed_2d_slice)
            emb_moving = self.image_encoder(moving_2d_slice)
        image_embedding_2d = torch.cat([emb_fixed, emb_moving], dim=1)
        
        sparse_emb, dense_emb = self.prompt_encoder(tokens=tokens)

        # 2. ViT-V-Net token features (将 moving 作为 source 通道，让 flow 把 moving warp 到 fixed)
        # ViTVNet.forward 里 warp 的 source = x[:, 0:1, ...]，因此这里 x_in 的第 0 通道必须是 moving
        x_in = torch.cat([moving_3d, fixed_3d], dim=1)
        source = x_in[:, 0:1, :, :, :]

        encoded_tokens, _, vit_features = self.register_model.transformer(x_in)
        # encoded_tokens: (B, n_patch, hidden) -> (B, hidden, l, h, w)
        b, n_patch, hidden = encoded_tokens.shape
        features_3d = encoded_tokens.permute(0, 2, 1).contiguous().view(b, hidden, self.l, self.h, self.w)

        # 3. GPR Fusion：用 MedSAM+文本 prompt 融合/替换 ViTVNet 的 bottleneck tokens 对应的 3D 特征
        original_shape = fixed_3d.shape[2:]
        slice_positions_batch = torch.stack([
            slice_ids.float(), 
            torch.full_like(slice_ids, original_shape[1], dtype=torch.float32), 
            torch.full_like(slice_ids, original_shape[2], dtype=torch.float32)
        ], dim=1).to(fixed_3d.device)

        fused_embedding = self.GPR(
            features_2d=image_embedding_2d,
            pos_2d=self.prompt_encoder.get_dense_pe(),
            sparse_code_2d=sparse_emb,
            dense_code_2d=dense_emb,
            features_3d=features_3d,
            slice_positions=slice_positions_batch,
            original_shape=original_shape
        )
        
        # 4. 把 (B, hidden, l, h, w) -> tokens 并喂给 decoder
        fused_tokens = fused_embedding.flatten(2).transpose(1, 2).contiguous()  # (B, n_patch, hidden)
        x_dec = self.register_model.decoder(fused_tokens, features=vit_features)
        flow = self.register_model.reg_head(x_dec)

        warped_moving = self.register_model.spatial_trans(source, flow)
        return warped_moving, flow

# ==========================================
# 6. 真实数据集加载 (IXI)
# ==========================================
def pkload(fname):
    with open(fname, 'rb') as f: return pickle.load(f)

class IXIDataset(Dataset):
    def __init__(self, data_path, atlas_path, img_size=(160, 192, 224), medsam_size=1024):
        self.paths = sorted(glob.glob(data_path))
        self.atlas_path = atlas_path
        self.img_size = img_size
        self.medsam_size = medsam_size
        self.tokenizer = CLIPTokenizer.from_pretrained("/home/suzixian/nas/VLM_space/registration/MedSAM-main/openai_clip-vit-base-patch16")
        
        self.label_dict = {
            1: "Cerebral White Matter", 2: "Cerebral Cortex", 3: "Lateral Ventricle",
            4: "Cerebellum White Matter", 5: "Cerebellum Cortex", 6: "Thalamus",
            7: "Caudate", 8: "Putamen", 9: "Pallidum", 10: "Hippocampus", 11: "Amygdala"
        }
        self.atlas_vol, self.atlas_seg = pkload(self.atlas_path)

    def __len__(self):
        return len(self.paths)

    def preprocess_2d(self, img_2d):
        img_resized = skimage_transform.resize(img_2d, (self.medsam_size, self.medsam_size), 
                                               order=3, preserve_range=True, mode='constant', anti_aliasing=True)
        img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
        img_3c = np.repeat(img_resized[np.newaxis, :, :], 3, axis=0)
        return torch.tensor(img_3c).float()

    def tokenize_text(self, text):
        return self.tokenizer(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze(0)

    def __getitem__(self, index):
        fixed_vol, fixed_seg = self.atlas_vol, self.atlas_seg
        moving_vol, moving_seg = pkload(self.paths[index])
        
        fixed_tensor = torch.from_numpy(fixed_vol).float().unsqueeze(0)
        moving_tensor = torch.from_numpy(moving_vol).float().unsqueeze(0)
        fixed_seg_tensor = torch.from_numpy(fixed_seg).long().unsqueeze(0)
        moving_seg_tensor = torch.from_numpy(moving_seg).long().unsqueeze(0)
        
        slice_id = random.randint(0, self.img_size[0] - 1)
        fixed_slice_tensor = self.preprocess_2d(fixed_vol[slice_id, :, :])
        moving_slice_tensor = self.preprocess_2d(moving_vol[slice_id, :, :])
        
        label_id = random.randint(1, 11)
        text_prompt = self.label_dict.get(label_id, "Brain")
        tokens = self.tokenize_text(text_prompt)
        
        return {
            "fixed_image": fixed_tensor, "moving_image": moving_tensor,
            "fixed_label": fixed_seg_tensor, "moving_label": moving_seg_tensor,
            "fixed_slice": fixed_slice_tensor, "moving_slice": moving_slice_tensor,
            "tokens": tokens, "slice_id": slice_id, "text_label_id": label_id
        }

def dice_score(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 2. * intersection / (union + 1e-5)

def log_images(writer, fixed, moving, warped, fixed_label, warped_label, epoch, tag="Train"):
    mid_slice = fixed.shape[2] // 2
    def norm_img(img):
        img = img.float().cpu().detach().numpy()
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    f_img = norm_img(fixed[0, 0, mid_slice, :, :])
    m_img = norm_img(moving[0, 0, mid_slice, :, :])
    w_img = norm_img(warped[0, 0, mid_slice, :, :])
    
    f_lbl = fixed_label[0, 0, mid_slice, :, :].cpu().detach().numpy()
    w_lbl = warped_label[0, 0, mid_slice, :, :].cpu().detach().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(f_img, cmap='gray'); axes[0].set_title("Fixed")
    axes[1].imshow(m_img, cmap='gray'); axes[1].set_title("Moving")
    axes[2].imshow(w_img, cmap='gray'); axes[2].set_title("Warped Moving")
    axes[3].imshow(f_lbl, cmap='tab20'); axes[3].set_title("Fixed Label")
    axes[4].imshow(w_lbl, cmap='tab20'); axes[4].set_title("Warped Label")
    for ax in axes: ax.axis('off')
    writer.add_figure(f"{tag}/Visuals", fig, epoch)
    plt.close(fig)

# ==========================================
# 7. Main Training Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/suzixian/nas/VLM_space/IXI_data/Train/*.pkl')
    parser.add_argument('--atlas_path', type=str, default='/home/suzixian/nas/VLM_space/IXI_data/atlas.pkl')
    parser.add_argument('--medsam_ckpt', type=str, default='work_dir/medsam_vit_b.pth')
    parser.add_argument('--work_dir', type=str, default='experiments/medsam_reg')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--weights', type=float, nargs='+', default=[1.0, 1.0, 0.5], help='Weights for NCC, Reg, Dice losses')
    args = parser.parse_args()
    
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.work_dir, 'logs'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Config & Data
    dataset = IXIDataset(args.data_dir, args.atlas_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    img_size = (160, 192, 224)
    config_vit = CONFIGS_ViT["ViT-V-Net"]
    
    # 2. Models
    medsam_model = sam_model_registry['vit_b'](checkpoint=args.medsam_ckpt)
    prompt_encoder = TextPromptEncoder(embed_dim=256, image_embedding_size=(64,64), input_image_size=(1024,1024), mask_in_chans=1)
    
    vit_model = ViTVNet(config_vit, img_size=img_size)
    bottleneck_channels = config_vit.hidden_size
    gpr = GeometricPerceptionReconstructionMR(
        input_dim=256,
        input_3d_channels=bottleneck_channels,
        output_3d_channels=bottleneck_channels,
        feature_size=(64, 64), 
        output_compressed_size=(5, 6, 7)
    )

    model = MedSAMRegNet(
        image_encoder=deepcopy(medsam_model.image_encoder),
        prompt_encoder=prompt_encoder,
        register_model=vit_model,
        gpr=gpr,
        img_size=img_size,
    ).to(device)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    sim_loss_fn = NCC_vxm()
    grad_loss_fn = Grad3d(penalty='l2')
    stn_label = SpatialTransformer((160, 192, 224), mode='nearest').to(device)
    
    global_step = 0
    best_loss = float('inf')
    
    print(f"Training started on {device}...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        pbar = tqdm(dataloader)
        
        for step, batch in enumerate(pbar):
            fixed = batch['fixed_image'].to(device)
            moving = batch['moving_image'].to(device)
            f_slice = batch['fixed_slice'].to(device)
            m_slice = batch['moving_slice'].to(device)
            tokens = batch['tokens'].to(device)
            slice_ids = batch['slice_id'].to(device)
            
            moving_seg = batch['moving_label'].to(device)
            fixed_seg = batch['fixed_label'].to(device)
            target_label_id = batch['text_label_id'].to(device)
            
            # Forward
            warped_moving, flow = model(fixed, moving, f_slice, m_slice, tokens, slice_ids)
            
            # Loss
            loss_ncc = sim_loss_fn(fixed, warped_moving)
            loss_reg = grad_loss_fn(flow) # Fixed call signature
            
            warped_seg = stn_label(moving_seg.float(), flow).long()
            loss_dice_batch = 0
            for i in range(args.batch_size):
                tid = target_label_id[i].item()
                pred_mask = (warped_seg[i] == tid).float()
                gt_mask = (fixed_seg[i] == tid).float()
                loss_dice_batch += (1 - dice_score(pred_mask, gt_mask))
            loss_dice = loss_dice_batch / args.batch_size
            
            w_ncc, w_reg, w_dice = args.weights
            total_loss = w_ncc * loss_ncc + w_reg * loss_reg + w_dice * loss_dice
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss.append(total_loss.item())
            writer.add_scalar('Train/Loss_Total', total_loss.item(), global_step)
            writer.add_scalar('Train/Loss_NCC', loss_ncc.item(), global_step)
            writer.add_scalar('Train/Loss_Reg', loss_reg.item(), global_step)
            writer.add_scalar('Train/Loss_Dice', loss_dice.item(), global_step)
            
            pbar.set_description(f"Ep {epoch} | Loss: {total_loss.item():.4f} | Dice: {1-loss_dice:.3f}")
            global_step += 1
            
            if global_step % 50 == 0:
                log_images(writer, fixed, moving, warped_moving, fixed_seg, warped_seg, global_step)

        # Save Checkpoint
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.work_dir, 'best_model.pth'))
            print(f"New best model saved at epoch {epoch} with loss {best_loss:.4f}")
            
        torch.save(model.state_dict(), os.path.join(args.work_dir, 'latest_model.pth'))

    writer.close()

if __name__ == "__main__":
    main()