import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import imageio
import os
import sys
import inspect
from tqdm import tqdm
from utils.metrices import *
import timm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from baselines.ViT.utils.metrices import *

from baselines.ViT.utils import render, seeder
from baselines.ViT.utils.saver import Saver
from baselines.ViT.utils.iou import IoU

from baselines.ViT.data.imagenet import Imagenet_Segmentation

from baselines.ViT.ViT_explanation_generator import Baselines, LRP, IG
# changed these two imports to match demo
from baselines.ViT.ViT_new import vit_base_patch16_224 as vit_for_cam, deit_base_distilled_patch16_224 as deit_for_cam
from baselines.ViT.ViT_LRP import deit_base_distilled_patch16_224, vit_base_patch16_224
from baselines.ViT.ViT_ig import vit_base_patch16_224 as vit_attr_rollout, deit_base_distilled_patch16_224 as deit_attr_rollout
from baselines.ViT.ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP


from baselines.ViT.DDS import denoise, attack, apply_dds

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch.nn.functional as F

plt.switch_backend('agg')

# hyperparameters
num_workers = 0
batch_size = 1

cls = ['airplane',
       'bicycle',
       'bird',
       'boat',
       'bottle',
       'bus',
       'car',
       'cat',
       'chair',
       'cow',
       'dining table',
       'dog',
       'horse',
       'motobike',
       'person',
       'potted plant',
       'sheep',
       'sofa',
       'train',
       'tv'
       ]

# Args
parser = argparse.ArgumentParser(description='Training multi-class classifier')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture')
parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                    help='Testing Dataset')
parser.add_argument('--method', type=str,
                    default='grad_rollout',
                    choices=['rollout', 'transformer_attribution', 'full_lrp', 'lrp_last_layer',
                             'attn_last_layer', 'attn_gradcam', 'attr_rollout'],
                    help='')
parser.add_argument('--thr', type=float, default=0.,
                    help='threshold')
parser.add_argument('--K', type=int, default=1,
                    help='new - top K results')
parser.add_argument('--save-img', action='store_true',
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
parser.add_argument('--with-dds', action='store_true',
                    default=False,
                    help='Use DDS')
parser.add_argument('--imagenet-seg-path', type=str, required=True)
parser.add_argument('--attack', action='store_true', default=False)
parser.add_argument('--attack_noise', type=float, default=8 / 255)
parser.add_argument('--seed', type=int, default=44)
parser.add_argument("--transformer", type=str, default="ViT", help='Currently supports ViT and DeiT')
parser.add_argument("--implementation-method", type=str, default="Hu", help="Hu or Chefer")
args = parser.parse_args()

args.checkname = args.method + '_' + args.arc

alpha = 2

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

seeder.seed_everything(args.seed)

# Define Saver
saver = Saver(args)
saver.results_dir = os.path.join(saver.experiment_dir, 'results')
if not os.path.exists(saver.results_dir):
    os.makedirs(saver.results_dir)
if not os.path.exists(os.path.join(saver.results_dir, 'input')):
    os.makedirs(os.path.join(saver.results_dir, 'input'))
if not os.path.exists(os.path.join(saver.results_dir, 'explain')):
    os.makedirs(os.path.join(saver.results_dir, 'explain'))

args.exp_img_path = os.path.join(saver.results_dir, 'explain/img')
if not os.path.exists(args.exp_img_path):
    os.makedirs(args.exp_img_path)
args.exp_np_path = os.path.join(saver.results_dir, 'explain/np')
if not os.path.exists(args.exp_np_path):
    os.makedirs(args.exp_np_path)

# Data
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
test_img_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])
test_lbl_trans = transforms.Compose([
    transforms.Resize((224, 224), Image.NEAREST),
])

ds = Imagenet_Segmentation(args.imagenet_seg_path,
                           transform=test_img_trans, target_transform=test_lbl_trans)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


if args.implementation_method.lower() == "hu":
    if args.method == 'attn_gradcam':
        if args.transformer.lower() == "vit":
            model = vit_for_cam(pretrained=True).to(device)
        elif args.transformer.lower() == "deit":
            model = deit_for_cam(pretrained=True).to(device)
    elif args.method == 'attr_rollout':
        if args.transformer.lower() == "vit":
            model = vit_attr_rollout(pretrained=True).to(device)
        elif args.transformer.lower() == "deit":
            model = deit_attr_rollout(pretrained=True).to(device)
    else:
        if args.transformer.lower() == "vit":
            model = vit_base_patch16_224(pretrained=True).to(device)
        elif args.transformer.lower() == "deit":
            model = deit_base_distilled_patch16_224(pretrained=True).to(device)
if args.implementation_method.lower() == "chefer":
    if args.method in ["full_lrp", "lrp_last_layer", "attn_last_layer"]:
        model = vit_orig_LRP(pretrained=True).to(device)
    elif args.method in ["rollout", "attn_gradcam"]:
        model = vit_for_cam(pretrained=True).to(device)
    elif args.method == "transformer_attribution":
        model = vit_base_patch16_224(pretrained=True).to(device)
else:
    # invalid implementation_method
    raise ValueError(f"Invalid implementation_method: {args.implementation_method}.")


model.eval()
print(model)

# Model
baselines = Baselines(model)

# LRP
lrp = LRP(model)

# attribution rollout
ig = IG(model)

metric = IoU(2, ignore_index=-1)

iterator = tqdm(dl)

model.eval()


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                    for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    # pred[0, 0] = 282
    # print('Pred cls : ' + str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = T.to(device)

    return Tt


def eval_batch(image, labels, evaluator, index):
    evaluator.zero_grad()
    # Save input image
    if args.save_img:
        img = image[0].permute(1, 2, 0).data.cpu().numpy()
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = img.astype('uint8')
        Image.fromarray(img, 'RGB').save(os.path.join(saver.results_dir, 'input/{}_input.png'.format(index)))
        Image.fromarray((labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8'), 'RGB').save(
            os.path.join(saver.results_dir, 'input/{}_mask.png'.format(index)))

    image.requires_grad = True

    image = image.requires_grad_()
    predictions = evaluator(image)
    attack_noise = args.attack_noise

    if args.attack:
        image = attack(image, model, attack_noise)

    if args.implementation_method.lower() == "hu":
        # segmentation test for the rollout baseline
        if args.method == 'rollout':
            def gen(image):
                return lrp.generate_LRP(image.to(device), method="rollout").reshape(batch_size, 1, 14, 14)

        # segmentation test for the LRP baseline (this is full LRP, not partial)
        elif args.method == 'full_lrp':
            def gen(image):
                return lrp.generate_LRP(image.to(device), method="full").reshape(batch_size, 1, 224, 224)

        # segmentation test for our method
        elif args.method == 'transformer_attribution':
            def gen(image):
                return (lrp.generate_LRP(image.to(device), start_layer=1, method="transformer_attribution")
                        .reshape(batch_size, 1, 14, 14))

        # segmentation test for the partial LRP baseline (last attn layer)
        elif args.method == 'lrp_last_layer':
            def gen(image):
                return (lrp.generate_LRP(image.to(device), method="last_layer", is_ablation=args.is_ablation)
                        .reshape(batch_size, 1, 14, 14))

        # segmentation test for the raw attention baseline (last attn layer)
        elif args.method == 'attn_last_layer':
            def gen(image):
                return (lrp.generate_LRP(image.to(device), method="last_layer_attn", is_ablation=args.is_ablation)
                        .reshape(batch_size, 1, 14, 14))

        # segmentation test for the GradCam baseline (last attn layer)
        elif args.method == 'attn_gradcam':
            # could be different look demo
            def gen(image):
                return baselines.generate_cam_attn(image.to(device)).reshape(batch_size, 1, 14, 14)

        elif args.method == 'attr_rollout':
            def gen(image):
                return (compute_rollout_attention(ig.generate_ig(image.cuda()))[:, 0, 1:]
                        .reshape(batch_size, 1, 14, 14))

        else:
            raise NotImplementedError(f'Method {args.method} not implemented')
    elif args.implementation_method.lower() == "chefer":
        def gen(image):
            if args.method == 'rollout':
                return baselines.generate_rollout(image.cuda(), start_layer=1).reshape(batch_size, 1, 14, 14)

            # segmentation test for the LRP baseline (this is full LRP, not partial)
            elif args.method == 'full_lrp':
                return lrp.generate_LRP(image.cuda(), method="full").reshape(batch_size, 1, 224, 224)

            # segmentation test for our method
            elif args.method == 'transformer_attribution':
                return lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution").reshape(
                    batch_size, 1, 14, 14)

            # segmentation test for the partial LRP baseline (last attn layer)
            elif args.method == 'lrp_last_layer':
                return lrp.generate_LRP(image.cuda(), method="last_layer", is_ablation=args.is_ablation) \
                    .reshape(batch_size, 1, 14, 14)

            # segmentation test for the raw attention baseline (last attn layer)
            elif args.method == 'attn_last_layer':
                return lrp.generate_LRP(image.cuda(), method="last_layer_attn", is_ablation=args.is_ablation) \
                    .reshape(batch_size, 1, 14, 14)

            # segmentation test for the GradCam baseline (last attn layer)
            elif args.method == 'attn_gradcam':
                return baselines.generate_cam_attn(image.cuda()).reshape(batch_size, 1, 14, 14)

            else:
                raise NotImplementedError(f"Method {args.method} not implemented")

    if args.with_dds:
        Res = apply_dds(image, args.attack, gen)
    else:
        Res = gen(image)

    if Res.shape[-1] != 224:
        # interpolate to full image size (224,224)
        Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').to(device)

    # threshold between FG and BG is the mean    
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    # I think this is necessary for segmentation eval, hence not in the demo
    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1 - Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    pred = Res.clamp(min=args.thr) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    if args.save_img:
        # Save predicted mask
        mask = F.interpolate(Res_1, [64, 64], mode='bilinear')
        mask = mask[0].squeeze().data.cpu().numpy()
        # mask = Res_1[0].squeeze().data.cpu().numpy()
        mask = 255 * mask
        mask = mask.astype('uint8')
        imageio.imsave(os.path.join(args.exp_img_path, 'mask_' + str(index) + '.jpg'), mask)

        relevance = F.interpolate(Res, [64, 64], mode='bilinear')
        relevance = relevance[0].permute(1, 2, 0).data.cpu().numpy()
        # relevance = Res[0].permute(1, 2, 0).data.cpu().numpy()
        hm = np.sum(relevance, axis=-1)
        maps = (render.hm_to_rgb(hm, scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
        imageio.imsave(os.path.join(args.exp_img_path, 'heatmap_' + str(index) + '.jpg'), maps)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation results
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
total_ap, total_f1 = [], []

predictions, targets = [], []
for batch_idx, (image, labels) in enumerate(iterator):

    if args.method == "blur":
        images = (image[0].to(device), image[1].to(device))
    else:
        images = image.to(device)
    labels = labels.to(device)

    correct, labeled, inter, union, ap, f1, pred, target = eval_batch(images, labels, model, batch_idx)

    predictions.append(pred)
    targets.append(target)

    total_correct += correct.astype('int64')
    total_label += labeled.astype('int64')
    total_inter += inter.astype('int64')
    total_union += union.astype('int64')
    total_ap += [ap]
    total_f1 += [f1]
    pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
    IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
    mIoU = IoU.mean()
    mAp = np.mean(total_ap)
    mF1 = np.mean(total_f1)
    iterator.set_description('pixAcc: %.4f, mIoU: %.4f, mAP: %.4f, mF1: %.4f' % (pixAcc, mIoU, mAp, mF1))

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

pr, rc, thr = precision_recall_curve(targets, predictions)
np.save(os.path.join(saver.experiment_dir, 'precision.npy'), pr)
np.save(os.path.join(saver.experiment_dir, 'recall.npy'), rc)

plt.figure()
plt.plot(rc, pr)
plt.savefig(os.path.join(saver.experiment_dir, 'PR_curve_{}.png'.format(args.method)))

txtfile = os.path.join(saver.experiment_dir, 'result_mIoU_%.4f.txt' % mIoU)
# txtfile = 'result_mIoU_%.4f.txt' % mIoU
fh = open(txtfile, 'w')
print("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
print("Mean AP over %d classes: %.4f\n" % (2, mAp))
print("Mean F1 over %d classes: %.4f\n" % (2, mF1))

fh.write("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
fh.write("Mean AP over %d classes: %.4f\n" % (2, mAp))
fh.write("Mean F1 over %d classes: %.4f\n" % (2, mF1))
fh.close()
