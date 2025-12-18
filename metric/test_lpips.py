import torch
import lpips

use_gpu = False  # Whether to use GPU
spatial = True  # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial)  # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

if (use_gpu):
    loss_fn.cuda()

## Example usage with dummy tensors
dummy_im0 = torch.zeros(1, 3, 256, 256)  # image should be RGB, normalized to [-1,1]
dummy_im1 = torch.zeros(1, 3, 256, 256)
if (use_gpu):
    dummy_im0 = dummy_im0.cuda()
    dummy_im1 = dummy_im1.cuda()
dist = loss_fn.forward(dummy_im0, dummy_im1)

## Example usage with images
ex_ref = lpips.im2tensor(lpips.load_image('/data/yjy_data/eval/pix2pix_S2O_new/realB/0599_real_B.png'))
ex_p0 = lpips.im2tensor(lpips.load_image('/data/yjy_data/eval/pix2pix_S2O_new/fakeB/0599_fake_B.png'))

if use_gpu:
    ex_ref = ex_ref.cuda()
    ex_p0 = ex_p0.cuda()

ex_d0 = loss_fn.forward(ex_ref, ex_p0)

if not spatial:
    print('Distances: (%.3f)' % ex_d0)
else:
    print(
        'Distances: (%.3f)' % (ex_d0.mean()))  # The mean distance is approximately the same as the non-spatial distance

    # Visualize a spatially-varying distance map between ex_p0 and ex_ref
    import matplotlib

    matplotlib.use("Agg")  # ① 一定放在最前面

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt

    plt.imshow(ex_d0[0, 0, ...].detach().cpu().numpy())
    plt.colorbar()
    plt.title("LPIPS map")
    plt.savefig("lpips_debug.png", dpi=150, bbox_inches="tight")
    plt.close()
