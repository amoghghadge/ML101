import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from ray import serve
from starlette.requests import Request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@serve.deployment(
    num_replicas=4,
    ray_actor_options={"num_gpus": 1}
)
@serve.ingress(app)
class MinecraftLLM:
    def __init__(self):
        """
        This method is called once per replica when the deployment starts.
        It now only loads, compiles, and warms up the fine-tuned model.
        """
        base_model_id = "google/gemma-3-12b-it"
        adapter_id = "amoghghadge/gemma-3-12b-mc-qa"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        print("Loading tokenizer and models...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        
        print(f"Loading LoRA adapter: {adapter_id}...")
        ft_model = PeftModel.from_pretrained(base_model, adapter_id)

        print("Compiling fine-tuned model...")
        self.model = torch.compile(ft_model, mode="reduce-overhead")
        self.model.eval()
        
        print("Performing warm-up call...")
        warmup_input = self.tokenizer("warmup", return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = self.model.generate(input_ids=warmup_input.input_ids, max_new_tokens=2)
        print("Server is warmed up and ready!")

    @app.post("/ask")
    async def ask_endpoint(self, http_request: Request) -> dict:
        json_payload = await http_request.json()
        question = json_payload["question"]

        prompt = f"Given the <USER_QUERY> about Minecraft, provide a helpful, accurate, and concise answer.\n\n<USER_QUERY>\n{question}\n</USER_QUERY>"
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        generation_kwargs = {"max_new_tokens": 256, "do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}

        with torch.no_grad():
            # --- THE FIX ---
            # Pass the `inputs` tensor directly to the `input_ids` argument.
            outputs = self.model.generate(input_ids=inputs, **generation_kwargs)

        # --- THE FIX FOR DECODING ---
        # The 'inputs' tensor now only has one dimension, so we use inputs.shape[-1]
        response_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

        return {"answer": response_text.strip()}

deployment = MinecraftLLM.bind()

