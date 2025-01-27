import os
from tqdm import tqdm
import h5py

import argparse

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 

# Import saliency methods and models
from misc_functions import *

from ViT_explanation_generator import Baselines, LRP

# from ViT_new import vit_base_patch16_224
# from ViT_LRP import vit_base_patch16_224 as vit_LRP

# changed these two imports to match demo
from baselines.ViT.ViT_new import vit_base_patch16_224 as vit_for_cam
from baselines.ViT.ViT_LRP import vit_base_patch16_224 

from ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP

from baselines.ViT.DDS import denoise, get_opt_t, trans_to_224, trans_to_256

from torchvision.datasets import ImageNet

from baselines.ViT.utils import seeder
from baselines.ViT.data.imagenet import create_balanced_dataset_subset
from collections import defaultdict
import pandas as pd


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

            if args.method == 'rollout':
                # Res = baselines.generate_rollout(data, start_layer=1).reshape(data.shape[0], 1, 14, 14)
                Res = lrp.generate_LRP(data, method="rollout").reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'lrp':
                Res = lrp.generate_LRP(data, start_layer=1, index=index).reshape(data.shape[0], 1, 14, 14)
                # no change?

            elif args.method == 'transformer_attribution':
                # Res = lrp.generate_LRP(data, start_layer=1, method="grad", index=index).reshape(data.shape[0], 1, 14, 14)
                Res = lrp.generate_LRP(data, method="transformer_attribution").reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'full_lrp':
                # Res = orig_lrp.generate_LRP(data, method="full", index=index).reshape(data.shape[0], 1, 224, 224)
                Res = lrp.generate_LRP(data, method="full").reshape(data.shape[0], 1, 224, 224)

            elif args.method == 'lrp_last_layer':
                # Res = orig_lrp.generate_LRP(data, method="last_layer", is_ablation=args.is_ablation, index=index) \
                #     .reshape(data.shape[0], 1, 14, 14)
                Res = lrp.generate_LRP(data, method="last_layer", is_ablation=args.is_ablation).reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'attn_last_layer':
                # Res = lrp.generate_LRP(data, method="last_layer_attn", is_ablation=args.is_ablation) \
                #     .reshape(data.shape[0], 1, 14, 14)
                 Res = lrp.generate_LRP(data, method="last_layer_attn", is_ablation=args.is_ablation).reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'attn_gradcam':
                Res = baselines.generate_cam_attn(data, index=index).reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'dds':
                # TODO hyperparams, possibly changed later
                m = 10
                res_list = []
                image = data
                for _ in range(m):
                    noise_level = 8 / 255
                    steps = 1000
                    start = 0.0001
                    end = 0.02
                    opt_t = get_opt_t(noise_level, start, end, steps)
                    image_noisy = image + torch.randn_like(image, ) * noise_level
                    image_dds = trans_to_224(denoise(trans_to_256(image_noisy), opt_t, steps, start, end, noise_level))
                    image_dds = torch.clamp(image_dds, -1, 1)
                    Res = lrp.generate_LRP(image_dds, start_layer=1, method="transformer_attribution", index=index).reshape(data.shape[0], 1, 14, 14)
                    res_list.append(Res)
                Res = torch.stack(res_list).mean(0)

            if args.method != 'full_lrp' and args.method != 'input_grads':
                Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
            
            Res = (Res - Res.min()) / (Res.max() - Res.min())

            data_cam[-data.shape[0]:] = Res.data.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                        choices=['rollout', 'lrp', 'transformer_attribution', 'full_lrp', 'lrp_last_layer',
                                 'attn_last_layer', 'attn_gradcam',
                                 'dds'],
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
    parser.add_argument('`--imagenet-subset-ratio`', type=float, default=1)
    args = parser.parse_args()

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.makedirs(os.path.join(PATH, 'visualizations'), exist_ok=True)

    try:
        os.remove(os.path.join(PATH, 'visualizations/{}/{}/results.hdf5'.format(args.method,
                                                                                args.vis_class)))
    except OSError:
        pass


    os.makedirs(os.path.join(PATH, 'visualizations/{}'.format(args.method)), exist_ok=True)
    if args.vis_class == 'index':
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                        args.vis_class,
                                                                        args.class_id)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                              args.vis_class,
                                                                              args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                     args.vis_class, ablation_fold)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                           args.vis_class, ablation_fold))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Model
    # model = vit_base_patch16_224(pretrained=True).cuda()
    # baselines = Baselines(model)
    model = vit_for_cam(pretrained=True).cuda()
    baselines = Baselines(model)

    # LRP
    # model_LRP = vit_LRP(pretrained=True).cuda()
    # model_LRP.eval()
    # lrp = LRP(model_LRP)
    model = vit_base_patch16_224(pretrained=True).cuda()
    lrp = LRP(model)

    # orig LRP
    model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
    model_orig_LRP.eval()
    orig_lrp = LRP(model_orig_LRP)

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
    # class_counts = defaultdict(int)
    # df_dict = dict()
    # for _, labels in sample_loader:
    #     for label in labels:
    #         class_counts[label.item()] += 1
    # for class_idx, count in class_counts.items():
    #     df_dict[imagenet_ds.classes[class_idx]] = count
    # print(df_dict)

    compute_saliency_and_save(args)
