import os
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig
from trl import SFTTrainer

# Explicitly set the device for each process to prevent memory spikes.
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

model_id = "google/gemma-3-12b-it"
dataset_id = "amoghghadge/gemma-3-12b-mc-qa-dataset"
new_model_id = "amoghghadge/gemma-3-12b-mc-qa"
model_class = AutoModelForCausalLM
torch_dtype = torch.bfloat16

dataset = load_dataset(dataset_id, split="train")

# Define model init arguments
model_kwargs = dict(
    attn_implementation="flash_attention_2", # Use "flash_attention_2" when running on Ampere or newer GPU
    dtype=torch_dtype, # What torch dtype to use, defaults to auto
    #device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch_dtype,
    #bnb_4bit_quant_storage=torch_dtype,
)

model = model_class.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    #modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
)

args = SFTConfig(
    output_dir="gemma-3-12b-mc-qa",         # directory to save and repository id
    hub_model_id=new_model_id,
    max_length=512,                         # max sequence length for model and packing of the dataset
    packing=True,                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=1,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=False,            # use gradient checkpointing to save memory
    optim="paged_adamw_8bit",              
    logging_steps=25,                       # log every 25 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-5,                     # learning rate, based on QLoRA paper
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    #ddp_find_unused_parameteres=True,
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    }
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer
)

trainer.train()

trainer.save_model()