# pokedec
This package can detect pokemon species based on images.

## Project Description
Pokémon are fascinating creatures beloved by millions worldwide, and their diversity makes them an exciting challenge for image classification. Detecting and classifying Pokémon images is not just a fun exercise, it also provides a valuable opportunity to explore and demonstrate simple machine learning techniques.

In this project, we aim to fine-tune a pre-trained model of the ResNet structure, which we found on [PyTorch Image Models (TIMM)](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file). This is called [ResNet50-D](https://huggingface.co/timm/resnet50d.ra4_e3600_r224_in1k), which was trained on the  ImageNet-1k dataset and has 25.6 million parameters. The new task will be to classify a Pokémon species from an image. Additionally, an important part of the project focuses on ensuring reproducibility and robustness and setting up an ML operations framework focusing on continuous integration.

We utilize the [“1000 Pokemon Dataset”](https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000?select=pokemon-dataset-1000) from Kaggle, which consists of 26,539 images of size 128 by 128 pixels comprising 1000 different Pokémon classes. The dataset is further split into 80% training, 10% validation and 10% testing. To ensure reliable classifications the data is augmented in order to generate more diverse training data. This is done using the TIMM framework or alternatively the built-in PyTorch augmentation functionality.

## Project structure
The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
