# Unsupervised Multi-Genre Music Generation

Deep generative models for symbolic music generation using LSTM autoencoders, VAEs, Transformers, and RLHF on the MAESTRO dataset.

## Project Structure

```
music-generation-unsupervised/
├── data/                    # Dataset and preprocessing output (not included)
├── notebooks/               # Jupyter notebooks for each task
│   ├── preprocessing_maestro.ipynb
│   ├── task1_autoencoder.ipynb
│   ├── task2_vae.ipynb
│   ├── task3_transformer.ipynb
│   └── task4_rlhf.ipynb
├── src/                     # Source code
│   ├── models/              # Model architectures
│   ├── training/            # Training scripts
│   ├── evaluation/          # Metrics computation
│   ├── preprocessing/       # MIDI parsing and tokenization
│   └── generation/          # MIDI export utilities
├── outputs/                 # Generated MIDI files and plots
└── requirements.txt         # Python dependencies
```

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `pretty_midi`, `numpy`, `matplotlib`, `pandas`, `scikit-learn`, `tqdm`, `openpyxl`

## Dataset

Download the **MAESTRO dataset v3.0.0** from [Google Magenta](https://magenta.tensorflow.org/datasets/maestro).

Update the dataset path in `notebooks/preprocessing_maestro.ipynb`:

## Usage

Run the notebooks in order:

1. `preprocessing_maestro.ipynb` – Convert MIDI to piano roll segments
2. `task1_autoencoder.ipynb` – Train LSTM autoencoder
3. `task2_vae.ipynb` – Train VAE (posterior collapse observed)
4. `task3_transformer.ipynb` – Train token-based transformer
5. `task4_rlhf.ipynb` – RLHF fine-tuning (requires human feedback Excel files)

## Results

| Model | Test Perplexity | Human Score (1-5) |
|-------|----------------|-------------------|
| LSTM Autoencoder | N/A | 0.97 |
| VAE (Task 2) | N/A | 3.87 |
| Transformer | 1.114 | 4.12 |
| RLHF (Final) | 1.104 | 4.51 |

## Important Notes

- **Dataset**: MAESTRO files are not included. Download separately and update the path.
- **RLHF Models**: Due to file size, trained RLHF checkpoints are not included. Run `task4_rlhf.ipynb` to generate samples from the RLHF-tuned model.
- **VAE**: The VAE exhibits posterior collapse as documented in the report. Code is provided for reference.

## Citation

If you use this code, please cite the MAESTRO dataset:

```bibtex
@inproceedings{hawthorne2019maestro,
  title={Enabling factorized piano music modeling and generation with the MAESTRO dataset},
  author={Hawthorne, Curtis and Stasyuk, Andriy and Roberts, Adam and Simon, Ian and Huang, Cheng-Zhi Anna and Dieleman, Sander and Elsen, Erich and Engel, Jesse and Eck, Douglas},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

## License

This project is for academic purposes as part of a course assignment.
