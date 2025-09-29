import dask.dataframe as dd
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    # 1. Load the raw dataset and the model's tokenizer
    print("Loading raw dataset and tokenizer...")
    dataset = load_dataset("naklecha/minecraft-question-answer-700k", split="train")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")

    # 2. Define the function to apply the chat template
    # This function creates the message structure and lets the tokenizer
    # convert it into a single, correctly formatted string.
    def apply_chat_template(sample):
        messages = [
            {"role": "user", "content": f"Given the <USER_QUERY> about Minecraft, provide a helpful, accurate, and concise answer.\n\n<USER_QUERY>\n{sample['question']}\n</USER_QUERY>"},
            {"role": "assistant", "content": sample["answer"]}
        ]
        # The key step: apply the template to get the final string
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # 3. Use Dask for parallel processing
    print("Converting to Dask DataFrame for parallel processing...")
    df = dataset.to_pandas()
    ddf = dd.from_pandas(df, npartitions=32) # Use 32 cores

    print("Applying chat template to all examples...")
    # Create a new 'text' column with the formatted chat strings
    ddf['text'] = ddf.apply(apply_chat_template, axis=1, meta=('text', 'str'))
    
    # Keep only the final text column
    final_ddf = ddf[['text']]

    # 4. Save the result to a local Parquet file
    output_path = "minecraft-qa-chat-templated.parquet"
    print(f"Saving processed data to {output_path}...")
    final_ddf.to_parquet(output_path, engine='pyarrow')

    # 5. Load the Parquet file and push to the Hub
    print("Loading from Parquet and pushing to Hugging Face Hub...")
    processed_dataset = load_dataset("parquet", data_files=f"{output_path}/*")
    
    hub_repo_id = "amoghghadge/gemma-3-12b-mc-qa-dataset"
    processed_dataset.push_to_hub(hub_repo_id)

    print(f"✅ Successfully created and published the dataset to {hub_repo_id}")

if __name__ == "__main__":
    main()