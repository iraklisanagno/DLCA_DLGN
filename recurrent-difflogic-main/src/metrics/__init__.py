import nltk

# Download required NLTK data silently
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
from metrics.base import (
    calculate_accuracy,
    calculate_perplexity,
    calculate_precision,
    calculate_recall,
    calculate_f1,
    calculate_top_5_accuracy,
    calculate_top_10_accuracy,
)

from metrics.machine_translation import (
    calculate_bleu,
    calculate_sacrebleu,
    calculate_meteor,
    calculate_nist,
    calculate_rouge_l,
    calculate_chrf,
)

METRIC_REGISTRY = {
    # Base metrics
    "accuracy": calculate_accuracy,
    "top_5_accuracy": calculate_top_5_accuracy,
    "top_10_accuracy": calculate_top_10_accuracy,
    "perplexity": calculate_perplexity,
    "precision": calculate_precision,
    "recall": calculate_recall,
    "f1": calculate_f1,
    
    # Machine translation metrics
    "bleu": calculate_bleu,
    "sacrebleu": calculate_sacrebleu,
    "meteor": calculate_meteor,
    "nist": calculate_nist,
    "rouge_l": calculate_rouge_l,
    "chrf": calculate_chrf,
}