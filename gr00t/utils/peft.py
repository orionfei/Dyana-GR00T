import torch
from peft import LoraConfig, get_peft_model


def _wrap_forward(model):
    def _forward(inputs):
        backbone_inputs, action_inputs = model.prepare_input(inputs)
        backbone_outputs = model.backbone(backbone_inputs)
        action_head_outputs = model.action_head(backbone_outputs, action_inputs)
        model.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    model.forward = _forward
    return model


def _get_modules_to_save(action_head_only: bool, tune_projector: bool) -> list[str] | None:
    if not tune_projector:
        return None

    prefix = "action_head."
    modules_to_save = [
        f"{prefix}state_encoder",
        f"{prefix}action_encoder",
        f"{prefix}action_decoder",
    ]

    # PEFT expects module names from the top-level model namespace.
    # The action head is always rooted at `action_head`, regardless of whether
    # LoRA targets only the action head or the full model.
    return modules_to_save


def get_lora_trainable_stats(model) -> dict[str, int]:
    lora_param_count = 0
    modules_to_save_param_count = 0
    total_trainable_param_count = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_count = int(parameter.numel())
        total_trainable_param_count += param_count
        if "lora_" in name:
            lora_param_count += param_count
        if "modules_to_save" in name:
            modules_to_save_param_count += param_count

    return {
        "lora_param_count": lora_param_count,
        "modules_to_save_param_count": modules_to_save_param_count,
        "total_trainable_param_count": total_trainable_param_count,
    }


def get_lora_model(
    model,
    rank=32,
    lora_alpha=16,
    lora_dropout=0.1,
    action_head_only=True,
    tune_projector=True,
    assert_lora_targets=False,
):
    target_modules = []

    # Inspect model structure to find the correct paths
    for name, module in model.named_modules():
        if action_head_only and "action_head" not in name:
            continue

        # Look for linear layers in attention mechanisms
        if isinstance(module, torch.nn.Linear):
            if any(x in name for x in ["q_proj", "v_proj", "to_q", "to_v", "k_proj", "to_k"]):
                target_modules.append(name)
    target_modules = sorted(set(target_modules))

    modules_to_save = _get_modules_to_save(
        action_head_only=action_head_only, tune_projector=tune_projector
    )

    print(f"LoRA target modules: {len(target_modules)} matched")
    if modules_to_save is not None:
        print(f"LoRA modules_to_save: {modules_to_save}")
    if assert_lora_targets and len(target_modules) == 0:
        raise ValueError(
            "LoRA target module count is 0. Refusing to continue because adapters would not be attached."
        )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    lora_stats = get_lora_trainable_stats(model)
    model._gr00t_lora_target_module_count = len(target_modules)
    model._gr00t_modules_to_save = tuple(modules_to_save or [])
    model._gr00t_lora_stats = lora_stats
    print(
        "LoRA trainable stats | "
        f"lora_params={lora_stats['lora_param_count']:,} "
        f"modules_to_save_params={lora_stats['modules_to_save_param_count']:,} "
        f"total_trainable_params={lora_stats['total_trainable_param_count']:,}"
    )
    if assert_lora_targets and lora_stats["lora_param_count"] <= 0:
        raise ValueError(
            "LoRA rank > 0 but no trainable LoRA parameters were found after PEFT wrapping."
        )

    model = _wrap_forward(model)

    return model
