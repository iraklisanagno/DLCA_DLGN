# Recurrent DiffLogic

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains the implementation for **Recurrent Deep Differential Logic Gate Networks (RDDLGNs)**, an extension of combinational DDLGNs ([Petersen et al., 2022](https://arxiv.org/abs/2210.08277)) introducing recurrent connections for stateful sequence modeling. This allows the framework to tackle a broader set of sequence learning tasks and benchmark against recurrent neural architectures.

---

## 🚀 Repository Structure

```
recurrent-difflogic
├── LICENSE                             # MIT License for the project
├── README.md                           # Project overview and instructions (this file)
├── configs/                            # Model and experiment configuration files for different datasets/tasks
├── data/                               # Folder for datasets and model checkpoints
├── logs/                               # Logs from runs (training, evaluation, sweeps)
├── notebooks/                          # Jupyter notebooks for result analysis, plotting, config editing
├── Singularity/                        # Singularity container definition and built image
│   └── pytorch2.4.0-cuda12.4-universal.def   # Environment definition (Python, CUDA, PyTorch etc.)
├── scripts/
│   └── run_config.sh                   # Example script to launch a config via sbatch
├── src/
│   ├── custom_datasets/                # Loaders for supported dataset types (WMT, translation, permutation, etc.)
│   │   ├── code_translation_dataloader.py
│   │   ├── permutation_dataloader.py
│   │   ├── rle_dataloader.py
│   │   ├── timeseries_dataloader.py
│   │   ├── video_dataloader.py
│   │   └── wmt_dataloader.py
│   ├── custom_tokenizers/              # Tokenizer implementations tailored to each dataset/task
│   │   ├── code_translation.py
│   │   ├── machine_translation.py
│   │   ├── permutation_tokenizer.py
│   │   └── rle_tokenizer.py
│   ├── main.py                         # Main entry point: training and evaluation pipeline
│   ├── metrics/                        # Metric computation utilities (BLEU, accuracy, etc.)
│   │   ├── base.py
│   │   └── machine_translation.py
│   ├── models/                         # Model definitions (RDDLGN, GRU, LSTM, Transformers, etc.)
│   │   ├── bidirectional_recurrent_difflogic.py
│   │   ├── bidirectional_unsynced_gru.py
│   │   ├── ...
│   │   ├── unsynced_recurrent_difflogic.py
│   │   └── ...
│   ├── trainer.py                      # Training loop and routines (logging, checkpointing, etc.)
│   ├── utils/                          # Utility functions (schedulers, wrappers, logic layers)
│   │   ├── channel_logic_layer.py
│   │   ├── gumbel_logic_wrapper.py
│   │   └── scheduler.py
│   └── visualization/                  # Visualization tools (wandb plots, prediction analysis)
│       ├── code_translation_visualization.py
│       ├── difflogic_visualizations.py
│       ├── permutation_accuracy_table.py
│       └── word_prediction.py
└── sweep/                              # Scripts and configs for hyperparameter sweeps with wandb
    ├── sweep_configs/                  # Sweep config files (e.g., lr.json, dropout.json)
    ├── run_sweep.sh                    # Script to launch sweep jobs via sbatch
    └── ...                             # Additional sweep utilities/scripts
```

*Each folder is commented with its purpose in the project. For more details, refer to the documentation and example configs/notebooks.*

---

## 🔧 Setup

**Reproducible Environment:**  
All experiments are conducted in a Singularity container with CUDA and PyTorch pre-installed for reproducibility across hardware/software setups.

1. **Clone repo:**  
   ```
   git clone https://gitlab.ethz.ch/disco-students/fs25/recurrent-difflogic.git
   ```
2. **Build the container:**  
   ```
   cd Singularity
   singularity build pytorch2.4.0-cuda12.4-universal.sif pytorch2.4.0-cuda12.4-universal.def
   ```
3. **Start the container:**
   ```
   singularity shell --nv pytorch2.4.0-cuda12.4-universal.sif
   ```

---

## 🧪 Experiments

All model and training configurations are stored in the `configs/` folder.  
To reproduce the experiments:

**Run manually:**  
   ```
   srun  --mem=25GB --gres=gpu:01 --exclude=tikgpu[06-10] --pty bash -i
   singularity exec --nv \
     --bind $(pwd):/mnt \
     --bind $(pwd)/data:/data \
     Singularity/pytorch2.4.0-cuda12.4-universal.sif \
     python /mnt/src/main.py --config /mnt/configs/wmt/unsynced_recurrent_difflogic.json
   ```

**Run with sbatch:**  
   ```
   sbatch scripts/run_config.sh  
   ```

**Run a sweep:**  
   - Create a sweep config inside `sweep/sweep_configs` (use `sweep/sweep_configs/lr.json` as a template).
   - Adjust sweep config path inside `sweep/run_sweep.sh` if needed.
   ```
   sbatch sweep/run_sweep.sh 
   ```

**Wandb integration:**  
On first run you need to enter your wandb API key.

**Bulk hyperparameter editing:**  
To efficiently change hyperparameters in all config files simultaneously, use:
`notebooks/config_bulk_editor.ipynb`

---

## 📊 Expected Results

- Results are logged to your wandb project folder (as defined in sweep or config).
- If run without sweep, results appear in a project folder called `RDDLGN`.
- Provided notebooks allow generation of tables, plots, and statistical analysis of experiments.

**Example plot:**  
`notebooks/complexity.ipynb` creates a scatterplot to compare model performance for different node counts:

![scatterplot](notebooks/nodes_accuracy_analysis/nodes_train_accuracy_loglinear_aaai_halfpage.png)

*Figure 1: Training accuracy vs. node count (log-scale, 0.01M to 50M) for GRU, RNN, and RDDLGN models. For RNN/GRU, node count is total parameters minus input embedding; for RDDLGN, it is the sum of all logic layer sizes. Log-linear fit slopes (R²): GRU 14.29 (0.98), RNN 10.34 (0.97), RDDLGN 13.31 (0.94).*

---

## 📄 License

This project uses the MIT License.  
See [LICENSE](LICENSE) for details.

---

## 📚 References

- Petersen et al., 2022: [Deep Differential Logic Gates](https://arxiv.org/abs/2210.08277)
- Ahmed et al., 2024: [WMT14 English-German Dataset](https://service.tib.eu/ldmservice/dataset/wmt-2014-english-to-german-dataset)