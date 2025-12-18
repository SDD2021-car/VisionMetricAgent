import os
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from metric.imqual_utils import getSSIM, getPSNR, getRMSE, getFSIM
import lpips
import cv2
import ssim
import torch
from skimage.metrics import structural_similarity
import logging
from image_similarity_measures import quality_metrics
import re

from datetime import datetime

def custom_sort_key(path):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


log_dir = "/data/yjy_data/eval"
os.makedirs(log_dir, exist_ok=True)

# 用时间戳区分每次调用
run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
log_file = os.path.join(log_dir, f"results_new_control_GF3_{run_id}.txt")

logger = logging.getLogger("metrics_agent")
logger.setLevel(logging.INFO)

# 避免重复添加 handler（重要！否则 import 多次会重复打印）
if not logger.handlers:
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger = logging.getLogger("metrics_agent")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize LPIPS model
loss_fn = lpips.LPIPS(net='alex', spatial=True)
loss_fn.cuda()
# def custom_sort_key(path):
#     filename = os.path.basename(path)
#     return (filename, len(filename))


# def SSIMs_PSNRs(dir, dir1, dir2, im_res=(256,256)):
#     """
#         - gtr_dir contain ground-truths
#         - gen_dir contain generated HR
#     """
#
#     gtr_paths = sorted(glob(os.path.join(dir1, "*.*")), key=custom_sort_key)
#     gen_paths = sorted(glob(os.path.join(dir2, "*.*")), key=custom_sort_key)
#
#     if len(gtr_paths) == 0 or len(gen_paths) == 0:
#         logger.warning(f"No images found in {dir1} or {dir2}")
#
#     ssims, psnrs, rmses, cw_ssims, lpips_result, fsims,sams = [], [], [], [], [], [],[]
#     # fsims = []
#     for gtr_path, gen_path in zip(gtr_paths, gen_paths):
#         logger.info(f"{gen_path}")  # 调试输出
#
#         # Read images
#         r_im = Image.open(gtr_path).resize(im_res)
#         g_im = Image.open(gen_path).resize(im_res)
#         r_im_rgb = r_im.convert("RGB")  # Convert to 3 channels
#         g_im_rgb = g_im.convert("RGB")  # Convert to 3 channels
#         # LPIPS computation
#         ex_ref = lpips.im2tensor(lpips.load_image(gtr_path))
#         ex_p0 = lpips.im2tensor(lpips.load_image(gen_path))
#         ex_ref = ex_ref.cuda()
#         ex_p0 = ex_p0.cuda()
#         ex_d0 = loss_fn.forward(ex_ref, ex_p0)
#         ex_d0 = ex_d0.cpu()
#         ex_d0_value = ex_d0.detach().numpy()
#         lpips_result.append(ex_d0_value.mean())
#
#         # Compute CW-SSIM
#         s = ssim.SSIM(r_im_rgb)
#         cw_ssims.append(s.cw_ssim_value(g_im_rgb))
#
#         # ssim1 = structural_similarity(np.array(r_im), np.array(g_im), data_range=255, channel_axis=2)
#         # ssims.append(ssim1)
#
#         # Compute FSIM
#         fsim = getFSIM(np.array(r_im_rgb), np.array(g_im_rgb))
#         fsims.append(fsim)
#
#     #     Compute RMSE
#         rmse = getRMSE(np.array(r_im_rgb), np.array(g_im_rgb))
#         rmses.append(rmse)
#     #
#     #     Compute PSNR
#         r_im = r_im.convert("L")
#         g_im = g_im.convert("L")
#         psnr = getPSNR(np.array(r_im_rgb), np.array(g_im_rgb))
#         logger.info(f"PSNR:{psnr}")
#         psnrs.append(psnr)
#     #
#     #     get SAM
#         sam = quality_metrics.sam(np.array(r_im_rgb, dtype=np.uint32), np.array(g_im_rgb, dtype=np.uint32))
#         # logger.info(f"SAM:{sam}")
#         sams.append(sam)
#     #
#     if len(ssims) == 0:
#         logger.warning(f"No valid SSIM values calculated for {dir}. Skipping...")
#
#     return np.array(ssims), np.array(psnrs), np.array(rmses), np.array(cw_ssims), np.array(lpips_result), np.array(fsims),np.array(sams)
#     # return np.array(fsims)
#
# # Initialize LPIPS model
# loss_fn = lpips.LPIPS(net='alex', spatial=True)
# loss_fn.cuda()

def SSIMs_PSNRs(gt_dir, gen_dir, im_res=(256, 256), device="cpu"):
    """
    gt_dir: ground-truth folder
    gen_dir: generated folder
    return: ssims, psnrs, rmses, cw_ssims, lpips_vals, fsims, sams  (all np.ndarray)
    """

    # 1) 收集文件并按“文件名”匹配（更稳）
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.*")), key=custom_sort_key)
    gen_paths = sorted(glob(os.path.join(gen_dir, "*.*")), key=custom_sort_key)

    if len(gt_paths) == 0 or len(gen_paths) == 0:
        logger.warning(f"No images found in {gt_dir} or {gen_dir}")
        return (np.array([]),) * 7

    gt_map = {os.path.basename(p): p for p in gt_paths}
    gen_map = {os.path.basename(p): p for p in gen_paths}
    common_names = sorted(set(gt_map.keys()) & set(gen_map.keys()), key=custom_sort_key)

    if len(common_names) == 0:
        logger.warning(f"No matched filenames between {gt_dir} and {gen_dir}")
        return (np.array([]),) * 7

    # 若有缺失，给个提示但不中断
    miss_gt = sorted(set(gen_map.keys()) - set(gt_map.keys()))
    miss_gen = sorted(set(gt_map.keys()) - set(gen_map.keys()))
    if miss_gt:
        logger.warning(f"{len(miss_gt)} files exist in gen but not in gt. Example: {miss_gt[:3]}")
    if miss_gen:
        logger.warning(f"{len(miss_gen)} files exist in gt but not in gen. Example: {miss_gen[:3]}")

    ssims, psnrs, rmses, cw_ssims, lpips_vals, fsims, sams = [], [], [], [], [], [], []

    # 2) LPIPS：只做推理，不建图
    # 确保 loss_fn 在外部已初始化；这里把它搬到 device
    if device and torch.cuda.is_available() and device.startswith("cuda"):
        dev = torch.device(device)
    else:
        dev = torch.device("cpu")

    # 3) resize 插值：建议 LANCZOS（更平滑），或 BICUBIC
    resample = Image.BICUBIC

    with torch.no_grad():
        for name in common_names:
            gtr_path = gt_map[name]
            gen_path = gen_map[name]

            # Read + resize
            r_rgb = Image.open(gtr_path).convert("RGB").resize(im_res, resample=resample)
            g_rgb = Image.open(gen_path).convert("RGB").resize(im_res, resample=resample)

            # ---- LPIPS（建议对 resize 后版本算，保证与其他指标同尺寸；否则会出现尺寸不一致）----
            # 如果你坚持用原图算 lpips，就把 r_rgb/g_rgb 保存到临时再 load_image；更慢。
            # 这里用 numpy->PIL->lpips 的 load_image 不方便，所以仍用路径，但会与 resize 后不一致。
            # 更稳的做法是：你改用 tensor 直接喂 LPIPS（推荐，但需要你把 loss_fn 的输入规范统一）。
            ex_ref = lpips.im2tensor(lpips.load_image(gtr_path))
            ex_p0 = lpips.im2tensor(lpips.load_image(gen_path))
            ex_ref = ex_ref.cuda()
            ex_p0 = ex_p0.cuda()
            ex_d0 = loss_fn.forward(ex_ref, ex_p0)
            ex_d0 = ex_d0.cpu()
            ex_d0_value = ex_d0.detach().numpy()
            lpips_vals.append(ex_d0_value.mean())

            # ---- CW-SSIM ----
            s = ssim.SSIM(r_rgb)
            cw_ssims.append(float(s.cw_ssim_value(g_rgb)))

            # ---- SSIM（RGB）----
            ssim_val = structural_similarity(
                np.array(r_rgb), np.array(g_rgb),
                data_range=255, channel_axis=2
            )
            ssims.append(float(ssim_val))

            # ---- FSIM（通常基于灰度/亮度，取决于你的 getFSIM 实现）----
            fsim_val = getFSIM(np.array(r_rgb), np.array(g_rgb))
            fsims.append(float(fsim_val))

            # ---- RMSE（RGB 或灰度取决于 getRMSE 实现）----
            rmse_val = getRMSE(np.array(r_rgb), np.array(g_rgb))
            rmses.append(float(rmse_val))

            # ---- PSNR（你原来用灰度；保留）----
            r_gray = r_rgb.convert("L")
            g_gray = g_rgb.convert("L")
            psnr_val = getPSNR(np.array(r_gray), np.array(g_gray))
            psnrs.append(float(psnr_val))

            # ---- SAM（建议用 RGB 而不是灰度）----
            sam_val = quality_metrics.sam(
                np.array(r_rgb, dtype=np.uint32),
                np.array(g_rgb, dtype=np.uint32)
            )
            sams.append(float(sam_val))

    if len(ssims) == 0:
        logger.warning(f"No valid values calculated for {gt_dir} vs {gen_dir}.")
        return (np.array([]),) * 7

    return (
        np.asarray(ssims),
        np.asarray(psnrs),
        np.asarray(rmses),
        np.asarray(cw_ssims),
        np.asarray(lpips_vals),
        np.asarray(fsims),
        np.asarray(sams),
    )

