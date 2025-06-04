# changed these two imports to match demo
from baselines.ViT.ViT_new import vit_base_patch16_224 as vit_for_cam, deit_base_distilled_patch16_224 as deit_for_cam
from baselines.ViT.ViT_LRP import deit_base_distilled_patch16_224, vit_base_patch16_224
from baselines.ViT.ViT_ig import vit_base_patch16_224 as vit_attr_rollout, \
    deit_base_distilled_patch16_224 as deit_attr_rollout
from baselines.ViT.ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP


def model_loader(implementation_method, method, transformer):
    if implementation_method.lower() == "hu":
        if method == 'attn_gradcam':
            if transformer.lower() == "vit":
                return vit_for_cam(pretrained=True)
            elif transformer.lower() == "deit":
                return deit_for_cam(pretrained=True)
        elif method == 'attr_rollout':
            if transformer.lower() == "vit":
                return vit_attr_rollout(pretrained=True)
            elif transformer.lower() == "deit":
                return deit_attr_rollout(pretrained=True)
        else:
            if transformer.lower() == "vit":
                return vit_base_patch16_224(pretrained=True)
            elif transformer.lower() == "deit":
                return deit_base_distilled_patch16_224(pretrained=True)
    elif implementation_method.lower() == "chefer":
        if method in ["full_lrp", "lrp_last_layer", "attn_last_layer"]:
            return vit_orig_LRP(pretrained=True)
        elif method in ["rollout", "attn_gradcam"]:
            return vit_for_cam(pretrained=True)
        elif method == "transformer_attribution":
            return vit_base_patch16_224(pretrained=True)
    else:
        raise ValueError(f"Invalid implementation_method: {implementation_method}.")
