# Sloths_Pain

A lightweight PyTorch Lightning project that classifies sloths vs pain au chocolat using deep learning. It includes data preparation, model training, evaluation reports, and an interactive Gradio demo.
There is two options to use our scripts.

## Quick Start
https://test.pypi.org/project/Sloths-Pain/

If you are running your own uv enviroment you can simply use the following commands to launch the demo.
```
uvx --from sloths-pain --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sloths-demo
```
You will see a link, if you visit it you can perform custom experiments in our demo enviroment.
The code is adapted to work with any two classes of images you want to work with. Feel free to create a directory with two inner folders, each containing the images of its own class (its name will be read in the folder's name) and use them for the experiment.
For a first controlled test, we suggest to use the intended database for our project in:
[https://www.kaggle.com/datasets/iamrahulthorat/sloths-versus-pain-au-chocolat](https://www.kaggle.com/datasets/iamrahulthorat/sloths-versus-pain-au-chocolat)

You can input you kaggle username and key obtained when creating a new api token. If you do so, the demo can download the dataset in a temporal file and run with the images located there automatically.

## Github

### Dataset

Download the dataset from Kaggle:
[https://www.kaggle.com/datasets/iamrahulthorat/sloths-versus-pain-au-chocolat](https://www.kaggle.com/datasets/iamrahulthorat/sloths-versus-pain-au-chocolat)

Organize the images as follows:
```
data/
├── sloths/
└── pain_au_chocolat/
```
### Execution

Train the model and generate reports:
```
uv run python src/sloths/module.py
```
Reports and CSV summaries will be created under:
```
reports/
├── figures/
├── class_distribution.csv
├── image_dimensions_summary.csv
└── train_validation_distribution.csv
```
### Interactive Demo

Run the interactive classifier interface:
```
uv run --active python -m sloths.demo.app_gradio
```
More about the demo in the Pypi section.

Then open the URL shown in the terminal.

### Testing
Run the unit tests:
```
uv run pytest -v
```



# Documentation
Full API reference is available at:
[https://odin-kaos.github.io/Sloths/](https://odin-kaos.github.io/Sloths/)

Authors: Odei & Lucas
Python: >=3.12
Frameworks: PyTorch, Lightning, Gradio
