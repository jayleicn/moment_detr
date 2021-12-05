import torch
from moment_detr.model import build_transformer, build_position_encoding, MomentDETR


def build_inference_model(ckpt_path, **kwargs):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["opt"]
    if len(kwargs) > 0:  # used to overwrite default args
        args.update(kwargs)
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = MomentDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
    )

    model.load_state_dict(ckpt["model"])
    return model


