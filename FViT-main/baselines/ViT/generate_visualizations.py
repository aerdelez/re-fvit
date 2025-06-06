import argparse
import inspect
import os
import sys

import h5py
import torch
from torchvision.datasets import ImageNet
from tqdm import tqdm

from baselines.ViT.DDS import apply_dds, attack
from baselines.ViT.ViT_explanation_generator import Baselines, LRP, IG, compute_rollout_attention
from baselines.ViT.data.imagenet import create_balanced_dataset_subset
from baselines.ViT.utils import seeder
from baselines.ViT.utils.model_loader import model_loader
from misc_functions import *

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)


def normalize(tensor,
              mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def compute_saliency_and_save(args):
    first = True
    with h5py.File(os.path.join(args.method_dir, 'results.hdf5'), 'a') as f:
        data_cam = f.create_dataset('vis',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")
        for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
            if first:
                first = False
                data_cam.resize(data_cam.shape[0] + data.shape[0] - 1, axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0] - 1, axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0] - 1, axis=0)
            else:
                data_cam.resize(data_cam.shape[0] + data.shape[0], axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0], axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0], axis=0)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            target = target.to(device)

            data = normalize(data)
            data = data.to(device)
            data.requires_grad_()

            index = None
            if args.vis_class == 'target':
                index = target

            # perturbation test for the rollout baseline
            if args.method == 'rollout':
                def gen(image):
                    return lrp.generate_LRP(image.to(device), method="rollout", start_layer=1).reshape(args.batch_size,
                                                                                                       1, 14, 14)

            # perturbation test for the LRP baseline (this is full LRP, not partial)
            elif args.method == 'full_lrp':
                def gen(image):
                    return lrp.generate_LRP(image.to(device), method="full", index=index, start_layer=1).reshape(
                        args.batch_size, 1, 224, 224)

            # perturbation test for our method
            elif args.method == 'transformer_attribution':
                def gen(image):
                    return (lrp.generate_LRP(image.to(device), start_layer=1, index=index,
                                             method="transformer_attribution").reshape(args.batch_size, 1, 14, 14))

            # perturbation test for the partial LRP baseline (last attn layer)
            elif args.method == 'lrp_last_layer':
                def gen(image):
                    return (lrp.generate_LRP(image.to(device), method="last_layer", is_ablation=args.is_ablation,
                                             index=index).reshape(args.batch_size, 1, 14, 14))

            # perturbation test for the raw attention baseline (last attn layer)
            elif args.method == 'attn_last_layer':
                def gen(image):
                    return (lrp.generate_LRP(image.to(device), method="last_layer_attn", is_ablation=args.is_ablation,
                                             index=index).reshape(args.batch_size, 1, 14, 14))

            # perturbation test for the GradCam baseline (last attn layer)
            elif args.method == 'attn_gradcam':
                # could be different look demo
                def gen(image):
                    return baselines.generate_cam_attn(image.to(device), index=index).reshape(args.batch_size, 1, 14,
                                                                                              14)

            elif args.method == 'attr_rollout':
                def gen(image):
                    return (
                        compute_rollout_attention(ig.generate_ig(image.cuda()))[:, 0, 1:].reshape(args.batch_size, 1,
                                                                                                  14, 14))
            else:
                raise NotImplementedError(f'Method {args.method} not implemented')

            # TODO attack before or after target
            if args.attack_noise > 0:
                args.attack = True
            if args.attack:
                if args.attack_noise >= 1:
                    data = attack(data, model, args.attack_noise / 255)
                else:
                    data = attack(data, model, args.attack_noise)

            if args.use_dds:
                # True is for attack
                Res = apply_dds(data, True, gen)
            else:
                Res = gen(data)

            if Res.shape[-1] != 224:
                # interpolate to full image size (224,224)
                Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').to(device)

            Res = (Res - Res.min()) / (Res.max() - Res.min())

            data_cam[-data.shape[0]:] = Res.data.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                        choices=['rollout', 'lrp', 'transformer_attribution', 'lrp_last_layer',
                                 'attn_last_layer', 'attn_gradcam', 'attr_rollout',
                                 #  'full_lrp',
                                 'dds_rollout', 'dds_lrp', 'dds_transformer_attribution', 'dds_lrp_last_layer',
                                 'dds_attn_last_layer', 'dds_attn_gradcam', 'dds_attr_rollout'],
                        help='')
    parser.add_argument('--lmd', type=float,
                        default=10,
                        help='')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='')
    parser.add_argument('--cls-agn', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-ia', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fgx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-m', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-reg', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--is-ablation', type=bool,
                        default=False,
                        help='')
    parser.add_argument('--imagenet-validation-path', type=str,
                        required=True,
                        help='')
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--imagenet-subset-ratio', type=float, default=1)
    parser.add_argument('--attack', action='store_true', default=False)
    parser.add_argument('--attack-noise', type=float, default=0)
    parser.add_argument("--transformer", type=str, default="ViT", help='Currently supports ViT and DeiT')
    parser.add_argument('--use-dds', action='store_true', default=False, help='Use DDS')
    args = parser.parse_args()

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.makedirs(os.path.join(PATH, 'visualizations'), exist_ok=True)

    dir_method = args.method
    if args.use_dds:
        dir_method = 'dds_' + dir_method
    try:
        os.remove(os.path.join(PATH, 'visualizations/{}/{}/results.hdf5'.format(dir_method, args.vis_class)))
    except OSError:
        pass

    os.makedirs(os.path.join(PATH, 'visualizations/{}'.format(dir_method)), exist_ok=True)
    if args.vis_class == 'index':
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}_{}_{}'.format(dir_method, args.vis_class, args.class_id,
                                                                           args.attack_noise)), exist_ok=True)
        args.method_dir = os.path.join(PATH,
                                       'visualizations/{}/{}_{}_{}'.format(dir_method, args.vis_class, args.class_id,
                                                                           args.attack_noise))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        os.makedirs(os.path.join(PATH,
                                 'visualizations/{}/{}_{}/{}'.format(dir_method, args.vis_class, args.attack_noise,
                                                                     ablation_fold)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}_{}/{}'.format(dir_method, args.vis_class,
                                                                                 args.attack_noise, ablation_fold))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model = model_loader(implementation_method="hu", method=args.method, transformer=args.transformer).to(device)

    # Model
    baselines = Baselines(model)

    # LRP
    lrp = LRP(model)

    # attribution rollout
    ig = IG(model)

    # Dataset loader for sample images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # download=False,
    imagenet_ds = ImageNet(args.imagenet_validation_path, split='val', transform=transform)

    seeder.seed_everything(args.seed)

    sampler = None
    if args.imagenet_subset_ratio < 1:
        sampler = create_balanced_dataset_subset(imagenet_ds, args.imagenet_subset_ratio)

    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        # FIXME check
        sampler=sampler
    )

    print(f"Imagenet size: {len(sample_loader)}")

    compute_saliency_and_save(args)
