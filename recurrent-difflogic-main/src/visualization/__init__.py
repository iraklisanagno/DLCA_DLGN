from .word_prediction import generate_tranlation_table
from .difflogic_visualizations import (
    logic_distribution_visualization,
    neuron_weight_histogram,
)
from .code_translation_visualization import (
    generate_code_translation_table,
    generate_code_translation_examples
)
from .permutation_accuracy_table import generate_permutation_accuracy_table

VISUALISATION_REGISTRY = {
    "translation_table": generate_tranlation_table,
    "logic_distribution": logic_distribution_visualization,
    "weight_histogram": neuron_weight_histogram,
    "code_translation_table": generate_code_translation_table,
    "code_translation_examples": generate_code_translation_examples,
    "permutation_accuracy_table": generate_permutation_accuracy_table,
}