#!/usr/bin/env python3
"""LoRA SFT scaffold for the ADTC 2026 agriculture submission (AgriDoc / Qwen2.5-1.5B).

Base : Qwen/Qwen2.5-1.5B-Instruct  (ungated, public — no HF access request needed)
Data : qwen_sft_mix.jsonl  (chat `messages`: distilled extension-officer STYLE +
       CALIBRATION block — ask/differential/refuse-dose/commit — + number-free anchors).
       Objective is BEHAVIOUR (how to reason + when to hedge), NOT fact memorisation,
       which is what made the earlier Llama fact/MCQ tune fabricate (DECISIONS DR-0019).
Loss : assistant-turns only when supported (ASSISTANT_ONLY_LOSS=1; Qwen uses ChatML).
Out  : LoRA adapter + merged fp16 model (then quantize to Q4_0 — quantize_to_q4_0.sh)
Gate : after quantizing, bare-chat the result vs base on held_out_eval.jsonl
       (bare_chat.py) — ship ONLY if clearly better with no new fabrication.

Version-robust: TRL renames args across releases (max_seq_length->max_length,
tokenizer->processing_class, etc.), so we introspect and pass only what's accepted.
Run in a GPU env. Deps: requirements.txt.   python train_lora.py
"""
import inspect
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATA   = os.environ.get("DATA", "qwen_sft_mix.jsonl")  # cwd by default (flat pod layout)
OUTDIR = os.environ.get("OUTDIR", "out/adapter")
MERGED = os.environ.get("MERGED", "out/merged")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1536"))  # longest differential answers ~450 tok
EPOCHS = float(os.environ.get("EPOCHS", "3"))             # small (~600 rows) behavioural set
LR = float(os.environ.get("LR", "1e-4"))                 # gentle: shape style, don't overwrite
# Assistant-only loss ON by default now (run1/run2 trained full-sequence — a real bug).
ASSISTANT_ONLY = os.environ.get("ASSISTANT_ONLY_LOSS", "1") == "1"

# ChatML template with {% generation %} markers so TRL masks the loss to ASSISTANT tokens
# only. Used for TRAINING ONLY — the shipped gguf keeps Qwen's original template (copied
# from the hub at merge time), so inference is unaffected.
QWEN_GEN_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}{% endgeneration %}"
    "{% else %}{{ message['content'] }}{% endif %}"
    "{{ '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    orig_chat_template = tok.chat_template          # keep for the shipped tokenizer
    if ASSISTANT_ONLY:
        tok.chat_template = QWEN_GEN_TEMPLATE       # training-only: enables assistant masking

    ds = load_dataset("json", data_files=DATA, split="train")  # rows: {"messages":[...]}

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    # ---- build SFTConfig with only the kwargs this TRL version accepts ----
    sft_params = set(inspect.signature(SFTConfig.__init__).parameters)
    # Llama-3.2 has a 128k vocab -> the CE logits tensor is (bsz*seq*128k); keep bsz
    # small and recover effective batch via accumulation. bsz=4 -> ~2GB logits spike.
    kw = dict(
        output_dir=OUTDIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=int(os.environ.get("BSZ", "4")),
        gradient_accumulation_steps=int(os.environ.get("ACC", "16")),   # eff. batch 64
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},   # needed with PEFT/LoRA
        learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.0, bf16=True, logging_steps=20, save_strategy="epoch",
        seed=42, report_to="none",
    )
    if "max_length" in sft_params:            # newer TRL
        kw["max_length"] = MAX_SEQ_LEN
    elif "max_seq_length" in sft_params:      # older TRL
        kw["max_seq_length"] = MAX_SEQ_LEN
    if "packing" in sft_params:
        kw["packing"] = False
    # assistant-only masking needs a chat template with {% generation %} markers, which
    # the stock Llama-3.2 template lacks. Off by default (full-sequence SFT — fine for
    # short QA/MCQ). Opt in with ASSISTANT_ONLY_LOSS=1 only if you supply a compatible template.
    if ASSISTANT_ONLY and "assistant_only_loss" in sft_params:
        kw["assistant_only_loss"] = True
    cfg = SFTConfig(**kw)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="bfloat16")

    # ---- SFTTrainer: tokenizer kwarg name also varies by version ----
    tr_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    tkw = dict(model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg)
    if "processing_class" in tr_params:       # newer transformers/TRL
        tkw["processing_class"] = tok
    elif "tokenizer" in tr_params:            # older
        tkw["tokenizer"] = tok
    trainer = SFTTrainer(**tkw)

    trainer.train()
    trainer.save_model(OUTDIR)                 # adapter

    merged = trainer.model.merge_and_unload()  # merge LoRA for GGUF conversion
    merged.save_pretrained(MERGED)
    tok.chat_template = orig_chat_template      # restore Qwen's real template for the gguf
    # Copy the ORIGINAL tokenizer files rather than re-serializing them: newer
    # transformers writes a tokenizer_class llama.cpp's converter can't load.
    from huggingface_hub import hf_hub_download
    import shutil
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        try:
            shutil.copy(hf_hub_download(BASE_MODEL, f), os.path.join(MERGED, f))
        except Exception:
            tok.save_pretrained(MERGED)  # fallback
    print(f"adapter -> {OUTDIR}\nmerged  -> {MERGED}\nnext: bash quantize_to_q4_0.sh {MERGED}")


if __name__ == "__main__":
    main()
