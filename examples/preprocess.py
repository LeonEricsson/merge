from merge.preprocessing.preprocess import prepare_dataset

"""
Datasets from HF need to be preprocessed before they can be used for training. The
`prepare_dataset` function is a wrapper around the preprocessing pipeline that
downloads, tokenizes, and splits sequences from a dataset. The function also
handles sequence length constraints and optionally pushes the processed dataset
to the HuggingFace Hub.
"""


def prepare_tinystories():
    prepare_dataset(
        dataset_name="roneneldan/TinyStories",
        tokenizer="EleutherAI/gpt-neo-125m",
        min_seq_length=256,
        max_seq_length=512,
        split_long_sequences=True,
        num_proc=16,
        split="train",
        push=False,
        hf_username="LeonEricsson",
    )


if __name__ == "__main__":
    prepare_tinystories()
