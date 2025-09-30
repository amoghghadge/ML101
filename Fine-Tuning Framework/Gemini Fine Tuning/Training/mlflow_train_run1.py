import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import mlflow # --- NEW --- Import the mlflow library

# --- NEW --- Configure MLFlow Tracking
# Set the experiment name
mlflow.set_experiment("Gemma 3 12B Minecraft Fine-Tuning")
# Set the tracking URI to the server you just started
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Explicitly set the device for each process
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

# --- The rest of your script is mostly the same ---
model_id = "google/gemma-3-12b-it"
dataset_id = "amoghghadge/gemma-3-12b-mc-qa-dataset" 
new_model_id = "amoghghadge/gemma-3-12b-mc-qa"

torch_dtype = torch.bfloat16
model_kwargs = dict(
    attn_implementation="flash_attention_2",
    dtype=torch_dtype,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch_dtype,
    )
)

model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

dataset = load_dataset(dataset_id, split="train")

args = SFTConfig(
    output_dir="gemma-3-12b-mc-qa-checkpoints", 
    hub_model_id=new_model_id,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    optim="paged_adamw_8bit",
    logging_steps=25,
    save_strategy="epoch",
    learning_rate=2e-5,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    push_to_hub=True,
    # --- CHANGED --- Tell the trainer to report to mlflow
    report_to="mlflow",
    dataset_text_field="text",
    max_length=512,
    packing=True,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

# Start a new MLFlow run
with mlflow.start_run():
    # Log hyperparameters that are not automatically logged by the trainer
    mlflow.log_param("lora_r", 8)
    mlflow.log_param("lora_alpha", 16)
    
    # Train the model
    trainer.train()

# The model is automatically saved to the Hub by the trainer.
# The Hugging Face MLflowCallback will also log the model as an artifact.