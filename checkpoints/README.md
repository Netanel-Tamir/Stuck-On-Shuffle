# Checkpoints

Six LoRA adapters, one per released model. Each folder contains

| file | content |
|---|---|
| `adapter_model.safetensors` | LoRA weights (rank 4, alpha 4) for `q_proj`, `k_proj`, `v_proj`, `o_proj` of the language decoder's attention layers |
| `adapter_config.json` | standard PEFT config; `base_model_name_or_path` is the Hugging Face hub id of the frozen base |
| `lines.json` | records that LiNeS layer scaling (gamma 0, ramp over the full decoder depth) has been **baked into `lora_B`** |
| `training_args.json` | the hyper-parameters of the finetuning run this adapter comes from |

All adapters are epoch-1 checkpoints of a one-epoch-equivalent finetuning on the
`sq_lp` split of the released triplet dataset (see the top-level README).

Because the LiNeS edit is already applied, load these with plain
`PeftModel.from_pretrained(base_model, folder)` and **do not** scale the layers
again. They are evaluation checkpoints: resuming training from them would train on
top of the scaled update. The wrappers in `stuck_on_shuffle/` do the right thing;
`load_model(name)` is the intended entry point.

| folder | base model | adapter size |
|---|---|---|
| `qwen3vl-embedding-8b` | Qwen/Qwen3-VL-Embedding-8B | 15 MB |
| `qwen3vl-embedding-2b` | Qwen/Qwen3-VL-Embedding-2B | 6 MB |
| `wemm-embedding-9b` | tencent/WeMM-Embedding-9B | 4 MB |
| `wemm-embedding-2b` | tencent/WeMM-Embedding-2B | 1.5 MB |
| `uembed-9b` | Alibaba-NLP/UEmbed-9B | 4 MB |
| `uembed-2b` | Alibaba-NLP/UEmbed-2B | 1.5 MB |

The WeMM and UEmbed backbones are hybrid stacks in which only every fourth decoder
layer uses softmax attention with `q/k/v/o` projections, so their adapters cover
6 (2B) or 8 (9B) layers. The adapter key prefix differs between the families
(UEmbed checkpoints are bare `Qwen3_5Model`s); the model classes in
`stuck_on_shuffle/` reproduce the exact module tree the adapters were trained on.
