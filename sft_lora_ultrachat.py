import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def to_single_turn_text(example):
    msgs = example["messages"]
    # Expect alternating user/assistant; take the first pair for stable SFT.
    user = next(m["content"] for m in msgs if m["role"] == "user")
    assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
    text = f"User: {user}\nAssistant: {assistant}"
    return {"text": text}

def main(
    base_model_dir="hf_pretrained",
    out_dir="hf_sft_lora",
    max_seq_len=512,
):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.map(to_single_turn_text, remove_columns=ds.column_names)

    tok = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"],  # GPT-2 style
    )
    model = get_peft_model(model, lora)

    # Train only on the assistant completion tokens
    collator = DataCollatorForCompletionOnlyLM(
        response_template="Assistant:",
        tokenizer=tok,
    )

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        bf16=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=3,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=args,
        max_seq_length=max_seq_len,
        data_collator=collator,
        packing=False,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
