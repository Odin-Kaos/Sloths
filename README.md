Sloths_Pain

A lightweight PyTorch Lightning project that classifies sloths vs pain au chocolat using deep learning. It includes data preparation, model training, evaluation reports, and an interactive Gradio demo.

Dataset

Download the dataset from Kaggle:
[https://www.kaggle.com/datasets/iamrahulthorat/sloths-versus-pain-au-chocolat](https://www.kaggle.com/datasets/iamrahulthorat/sloths-versus-pain-au-chocolat)

Organize the images as follows:

data/
├── sloths/
└── pain_au_chocolat/


Quick Start

Train the model and generate reports:

uv run python src/sloths/module.py

Reports and CSV summaries will be created under:

reports/
├── figures/
├── class_distribution.csv
├── image_dimensions_summary.csv
└── train_validation_distribution.csv
```

Gradio Demo

Run the interactive classifier interface:

uv run --active python -m sloths.demo.app_gradio

Then open the URL shown in the terminal.

Documentation
Full API reference is available at:
[https://odin-kaos.github.io/Sloths/](https://odin-kaos.github.io/Sloths/)

Testing
Run the unit tests:

uv run pytest -v

Authors: Odei & Lucas
Python: >=3.12
Frameworks: PyTorch, Lightning, Gradio
