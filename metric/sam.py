## python libs
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from metric.imqual_utils import getSSIM, getPSNR
from image_similarity_measures import quality_metrics
import cv2


## compares avg ssim and psnr
def evaluation(dir, im_res=(512, 512)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated HR
    """
    dir1 = '/data/yjy_data/Pix2Pix_S2O/result/experiment_pix_SAR2OPT_0111/test_200/fakeB'
    dir2 = '/data/yjy_data/Pix2Pix_S2O/result/experiment_pix_SAR2OPT_0111/test_200/realB'
    gtr_paths = sorted(glob(join(dir1, "*.*")))
    gen_paths = sorted(glob(join(dir2, "*.*")))
    ssims, psnrs, sams, rmses, fsims, issms, sres, uiqs, psnr2s, ssim2s = [], [], [], [], [], [], [], [], [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]

        # read HR from two datasets
        r_im = Image.open(gtr_path).resize(im_res)
        g_im = Image.open(gen_path).resize(im_res)

        # get SAM
        sam = quality_metrics.sam(np.array(r_im, dtype=np.uint32), np.array(g_im, dtype=np.uint32))
        sams.append(sam)
        # get RMSE
        # rmse = quality_metrics.rmse(np.array(r_im), np.array(g_im))
        # rmses.append(rmse)
        # # get FSIM
        # fsim = quality_metrics.fsim(np.array(r_im), np.array(g_im))
        # fsims.append(fsim)
        # # get ISSM
        # issm = quality_metrics.issm(np.array(r_im), np.array(g_im))
        # issms.append(issm)
        # # get SRE
        # sre = quality_metrics.sre(np.array(r_im), np.array(g_im))
        # sres.append(sre)
        # # get UIQ
        # uiq = quality_metrics.uiq(np.array(r_im), np.array(g_im))
        # uiqs.append(uiq)
        # # get PSNR test
        # psnr2 = quality_metrics.psnr(np.array(r_im), np.array(g_im))
        # psnr2s.append(psnr2)
        # # get SSIM test
        # ssim2 = quality_metrics.ssim(np.array(r_im), np.array(g_im))
        # ssim2s.append(ssim2)
        # # get ssim on RGB channels
        # ssim = getSSIM(np.array(r_im), np.array(g_im))
        # ssims.append(ssim)
        # # get psnt on L channel (SOTA norm)
        # r_im = r_im.convert("L")
        # g_im = g_im.convert("L")
        # psnr = getPSNR(np.array(r_im), np.array(g_im))
        # psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs), np.array(sams), np.array(rmses), np.array(fsims), np.array(issms), np.array(sres), np.array(uiqs)
        #, np.array(psnr2s), np.array(ssim2s)


"""
Get datasets from
 - http://irvlab.cs.umn.edu/resources/euvp-dataset
 - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
"""
dir = "/data/yjy_data/pix2pix_and_cyclegan_0808/result_SAR2OPT/result1/experiment_s_cyc_SAR2OPT_300/test_latest"

### compute SSIM and PSNR
SSIM_measures, PSNR_measures, SAM_measures, RMSE_measures, FSIM_measures, ISSM_measures, SRE_measures, UIQ_measures = evaluation(dir)
#, PSNR2_measures, SSIM2_measures
print("SSIM on {0} samples".format(len(SSIM_measures)))
print("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))

print("PSNR on {0} samples".format(len(PSNR_measures)))
print("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))

print("SAM on {0} samples".format(len(SAM_measures)))
print("Mean: {0} std: {1}".format(np.mean(SAM_measures), np.std(SAM_measures)))
print("MAX: {0} MIN:{1}".format(np.max(SAM_measures),np.min(SAM_measures)))

print("RMSE on {0} samples".format(len(RMSE_measures)))
print("Mean: {0} std: {1}".format(np.mean(RMSE_measures), np.std(RMSE_measures)))

print("FSIM on {0} samples".format(len(FSIM_measures)))
print("Mean: {0} std: {1}".format(np.mean(FSIM_measures), np.std(FSIM_measures)))

print("ISSM on {0} samples".format(len(ISSM_measures)))
print("Mean: {0} std: {1}".format(np.mean(ISSM_measures), np.std(ISSM_measures)))

print("SRE on {0} samples".format(len(SRE_measures)))
print("Mean: {0} std: {1}".format(np.mean(SRE_measures), np.std(SRE_measures)))

print("UIQ on {0} samples".format(len(UIQ_measures)))
print("Mean: {0} std: {1}".format(np.mean(UIQ_measures), np.std(UIQ_measures)))
"""
print("PSNR2 on {0} samples".format(len(PSNR2_measures)))
print("Mean: {0} std: {1}".format(np.mean(PSNR2_measures), np.std(PSNR2_measures)))

print("SSIM2 on {0} samples".format(len(SSIM2_measures)))
print("Mean: {0} std: {1}".format(np.mean(SSIM2_measures), np.std(SSIM2_measures)))
"""