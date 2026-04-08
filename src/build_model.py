from model.resnet_proto import build_resnet_proto


MODEL_BUILDERS = {
    "resnet_proto": build_resnet_proto,
}


def build_model(cfg):
    model_cfg = cfg.model
    model_type = model_cfg.type
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = MODEL_BUILDERS[model_type](
        model_cfg.depth,
        vf_dim=model_cfg.vf_dim,
        num_prototypes=model_cfg.num_prototypes,
        proj_use_layernorm=model_cfg.proj_use_layernorm,
    )

    model.load_prototypes_from_joblib(model_cfg.prototype_model_path)
    return model
