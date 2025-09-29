import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import mlflow

# --- FIX --- Only the main process (rank 0) sets up the experiment
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    mlflow.set_experiment("Gemma 3 12B Minecraft Fine-Tuning")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set device for each process
torch.cuda.set_device(local_rank)

# Define Model and Dataset IDs
model_id = "google/gemma-3-12b-it"
dataset_id = "amoghghadge/gemma-3-12b-mc-qa-dataset" 
new_model_id = "amoghghadge/gemma-3-12b-mc-qa-2" # Changed name for new model

# Configure Quantization and Model Arguments
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

# Load Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Configure PEFT (LoRA)
peft_config = LoraConfig(
    # --- TUNED --- Increased alpha and rank for more model capacity
    lora_alpha=64,
    lora_dropout=0.05,
    r=32,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Load Dataset
dataset = load_dataset(dataset_id, split="train")

# Define Training Arguments
args = SFTConfig(
    output_dir="gemma-3-12b-mc-qa-tuned-checkpoints", 
    hub_model_id=new_model_id,
    # --- TUNED --- Increased epochs for more training time
    num_train_epochs=3,
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
    report_to="mlflow",
    dataset_text_field="text",
    max_length=512,
    packing=True,
)

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer, # Corrected based on your working script
)

# --- FIX --- Start a single MLFlow run only on the main process
with mlflow.start_run(run_name=f"r_{peft_config.r}_alpha_{peft_config.lora_alpha}_epochs_{args.num_train_epochs}"):
    if local_rank == 0:
        # Log hyperparameters
        mlflow.log_params({
            "model_id": model_id,
            "dataset_id": dataset_id,
            "lora_r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "learning_rate": args.learning_rate,
            "epochs": args.num_train_epochs,
        })
    
    # Train the model
    trainer.train()

# The model is automatically saved to the Hub by the trainer.