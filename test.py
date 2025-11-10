
import SimpleITK as sitk
import numpy as np
import os
import glob
from os.path import join, basename
# def find_label_file(image_path, label_dir):
#     """
#     根据 image 文件路径找到对应的 label 文件路径。
#     Args:
#         image_path (str): image 文件路径。
#         label_dir (str): label 文件所在的目录。
#     Returns:
#         str: label 文件路径，如果找不到则返回 None。
#     """
#     label_filename = "D:\Desktop\SAM\MedSAM\data\FLARE22Train\labels\\" + "FLARE22_Tr_"+image_path[-16:-12]+".nii.gz"
#     return label_filename if os.path.exists(label_filename) else None
def resize_image_to_target_shape(image, target_shape):
    """
    将图像调整为指定的目标 shape。
    Args:
        image (SimpleITK.Image): 输入的 SimpleITK 图像。
        target_shape (tuple): 目标 shape，例如 (128, 512, 512)。
    Returns:
        SimpleITK.Image: 调整后的图像。
    """
    # 将 SimpleITK 图像转换为 NumPy 数组
    array = sitk.GetArrayFromImage(image)
    
    # 获取当前图像的 shape
    current_shape = array.shape
    
    # 在第一维度上进行填充或裁剪
    if current_shape[0] < target_shape[0]:
        # 填充
        pad_size = target_shape[0] - current_shape[0]
        pad_before = pad_size // 2
        pad_after = pad_size - pad_before
        array = np.pad(array, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
    elif current_shape[0] > target_shape[0]:
        # 裁剪
        crop_start = (current_shape[0] - target_shape[0]) // 2
        crop_end = crop_start + target_shape[0]
        array = array[crop_start:crop_end, :, :]
    
    # 将 NumPy 数组转换回 SimpleITK 图像
    resized_image = sitk.GetImageFromArray(array)
    
    # 保留原始图像的 spacing 和 direction 信息
    resized_image.SetSpacing(image.GetSpacing())
    resized_image.SetDirection(image.GetDirection())
    resized_image.SetOrigin(image.GetOrigin())
    
    return resized_image

def main(image_dir, target_shape, output_dir):
    """
    主函数：将目录下所有 .nii.gz 文件调整为指定的 target_shape，并处理对应的 label 文件。
    Args:
        image_dir (str): image 文件所在的目录。
        label_dir (str): label 文件所在的目录。
        target_shape (tuple): 目标 shape，例如 (128, 512, 512)。
        output_dir (str): 处理后的文件保存目录。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 .nii.gz 文件
    nii_gz_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".nii.gz")]
    
    if not nii_gz_files:
        print(f"No .nii.gz files found in {image_dir}")
        return
    
    # 处理每个文件
    for image_path in nii_gz_files:
        try:
            # 找到对应的 label 文件
            # label_path = find_label_file(image_path, label_dir)
            # if not label_path:
            #     print(f"No corresponding label file found for {image_path}")
            #     continue
            
            # 读取 image 和 label 文件
            image = sitk.ReadImage(image_path)
            # label = sitk.ReadImage(label_path)
            
            # 调整 image 和 label 的大小
            resized_image = resize_image_to_target_shape(image, target_shape)
            # resized_label = resize_image_to_target_shape(label, target_shape)
            
            # 保存处理后的文件
            image_filename = os.path.basename(image_path)
            # label_filename = os.path.basename(label_path)
            
            output_image_path = os.path.join(output_dir, image_filename)
            # output_label_path = os.path.join(output_dir, label_filename.replace(".nii.gz", "_resized.nii.gz"))
            
            sitk.WriteImage(resized_image, output_image_path)
            # sitk.WriteImage(resized_label, output_label_path)
            
            print(f"Resized {image_path} to {target_shape}, saved as {output_image_path} ")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    


directory =  # 替换为你的目录路径
# label_dir =   # label 文件目录

target_shape = (128, 512, 512)  # 目标 shape
output_dir =  # 处理后的文件保存目录
    
# 运行主函数
main(directory, target_shape, output_dir)
