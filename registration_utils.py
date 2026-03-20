import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SpatialTransformer(nn.Module):
    """N-D spatial transformer: 将 flow 作用到 src，并支持 bilinear / nearest warping。"""

    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0).type(torch.FloatTensor)
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


def dice_score(pred, target):
    """pred/target: (D,H,W) float mask, 返回 Dice 系数。"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 2.0 * intersection / (union + 1e-5)


def log_images(writer, fixed, moving, warped, fixed_label, warped_label, epoch, tag="Train"):
    """TensorBoard 可视化：展示固定/移动/配准后图像与标签（取中间切片）。"""

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
    axes[0].imshow(f_img, cmap="gray")
    axes[0].set_title("Fixed")
    axes[1].imshow(m_img, cmap="gray")
    axes[1].set_title("Moving")
    axes[2].imshow(w_img, cmap="gray")
    axes[2].set_title("Warped Moving")
    axes[3].imshow(f_lbl, cmap="tab20")
    axes[3].set_title("Fixed Label")
    axes[4].imshow(w_lbl, cmap="tab20")
    axes[4].set_title("Warped Label")
    for ax in axes:
        ax.axis("off")

    writer.add_figure(f"{tag}/Visuals", fig, epoch)
    plt.close(fig)

