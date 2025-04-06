import sys

import torch

sys.path.append("./guided-diffusion")
import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision.transforms as transforms
import torchattacks

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    # model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="250",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=256,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=4,
        use_ddim=False,
        model_path="./guided-diffusion/models/256x256_diffusion_uncond.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


args = create_argparser().parse_args([])

dist_util.setup_dist()
logger.configure()

logger.log("creating model and diffusion...")
d_model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
d_model.load_state_dict(
    dist_util.load_state_dict(args.model_path, map_location="cpu")
)
d_model.to(dist_util.dev())
if args.use_fp16:
    d_model.convert_to_fp16()
d_model.eval()
device = next(d_model.parameters()).device

shape = (1, 3, 256, 256)
steps = 1000
start = 0.0001
end = 0.02


def get_opt_t(delta, start, end, steps):
    return np.clip(int(np.around(1 + (steps - 1) / (end - start) * (1 - 1 / (1 + delta ** 2) - start))), 0, steps)


def beta(t, steps, start, end):
    return (t - 1) / (steps - 1) * (end - start) + start


def add_noise(x, delta, opt_t, steps, start, end):
    return np.sqrt(1 - beta(opt_t, steps, start, end)) * (x + th.randn_like(x) * delta)


def denoise(img, opt_t, steps, start, end, delta, direct_pred=False):
    img_xt = add_noise(img, delta, opt_t, steps, start, end).to(device)

    indices = list(range(opt_t))[::-1]
    img_iter = img_xt
    for i in indices:
        t = th.tensor([i] * shape[0], device=device)
        # t = t.to(device)
        with th.no_grad():
            out = diffusion.p_sample(
                d_model,
                img_iter,
                t,
                clip_denoised=args.clip_denoised,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs={},
            )
            img_iter = out['sample']
            if direct_pred:
                return out['pred_xstart']
    # img_iter = ((img_iter + 1) * 127.5).clamp(0, 255).to(th.uint8)
    # img_iter = img_iter.permute(0, 2, 3, 1)
    # img_iter = img_iter.contiguous()
    return img_iter


def range_of_delta(beta_s, beta_e, steps):
    def delta_value(beta):
        return (beta / (1 - beta)) ** (0.5)

    return (delta_value(beta_s), delta_value(beta_e))


trans_to_256 = transforms.Compose([
    transforms.Resize((256, 256)), ])
trans_to_224 = transforms.Compose([
    transforms.Resize((224, 224)), ])
delta_range = range_of_delta(start, end, steps)


def apply_dds(image,
              is_attacked,
              callback=None,
              noise_level=8 / 255,
              steps=1000,
              start=0.0001,
              end=0.02):
    # m = 2 if is_attacked else 10
    # TODO
    m=5
    opt_t = get_opt_t(noise_level, start, end, steps)

    image_noisy = image + torch.randn_like(image, ) * noise_level
    image_dds = trans_to_224(denoise(trans_to_256(image_noisy), opt_t, steps, start, end, noise_level))
    image_dds = torch.clamp(image_dds, -1, 1)

    Res = None
    if callback is not None:
        res_list = []
        for _ in range(m):
            Res = callback(image_dds)
            res_list.append(Res)
        Res = torch.stack(res_list).mean(0)

    # print('end dds')
    return image_dds, Res


def attack(image, model, noise_level, label_index=None, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    th.backends.cudnn.deterministic = True
    atk = torchattacks.PGD(model, eps=noise_level, alpha=noise_level / 5, steps=10)
    atk.set_normalization_used(mean, std)
    labels = th.FloatTensor([0] * 1000)
    if label_index == None:
        # with torch.no_grad():
        logits = model(image)
        label_index = logits.argmax()
        # print(label_index)

    labels[label_index] = 1
    labels = labels.reshape(1, 1000)
    adv_images = atk(image, labels.float())
    return adv_images
