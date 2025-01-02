from merge.preprocessing.preprocess import prepare_dataset
from merge.preprocessing.tokenizer import AutoTokenizer
"""
Datasets from HF need to be preprocessed before they can be used for training. The
`prepare_dataset` function is a wrapper around the preprocessing pipeline that
downloads, tokenizes, and splits sequences from a dataset. The function also
handles sequence length constraints and optionally pushes the processed dataset
to the HuggingFace Hub.
"""


def prepare_tinystories():
    tokenizer = AutoTokenizer.create("EleutherAI/gpt-neo-125m")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    prepare_dataset(
        dataset_name="roneneldan/TinyStories",
        tokenizer=tokenizer,
        min_seq_length=256,
        max_seq_length=512,
        split_long_sequences=True,
        num_proc=8,
        split="train",
        push=False,
        hf_username="LeonEricsson",
    )


if __name__ == "__main__":
    prepare_tinystories()
