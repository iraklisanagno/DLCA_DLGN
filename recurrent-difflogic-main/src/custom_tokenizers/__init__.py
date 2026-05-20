from custom_tokenizers.machine_translation import mt_tokenizer
from custom_tokenizers.rle_tokenizer import rle_tokenizer_pipeline
from custom_tokenizers.code_translation import code_translation_tokenizer
from custom_tokenizers.permutation_tokenizer import permutation_tokenizer_pipeline

TOKENIZER_REGISTRY = {
    "machine_translation": mt_tokenizer,
    "rle": rle_tokenizer_pipeline,
    "code_translation": code_translation_tokenizer,
    "permutation": permutation_tokenizer_pipeline
}