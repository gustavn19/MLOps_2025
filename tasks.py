import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "pokedec"
PYTHON_VERSION = "3.12"


@task
def python(ctx):
    """ """
    ctx.run(
        "which python" if os.name != "nt" else "where python",
        echo=True,
        pty=not WINDOWS,
    )


@task
def git(ctx, message):
    ctx.run("git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")


@task
def push_all(ctx, message):
    # freeze requirements and update requirements.txt
    ctx.run("pip freeze > requirements.txt")
    ctx.run("git add requirements.txt")
    ctx.run("git commit -m 'update requirements.txt'")
    ctx.run("git add .")
    ctx.run("git commit -m '" + message + "'")
    ctx.run("git push")


@task
def dvc(ctx, folder: str = "data", message: str = "Add_new_data"):
    ctx.run(f"dvc add {folder}")
    ctx.run(f"git add {folder}.dvc .gitignore")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")
    ctx.run("dvc push")


@task
def pull_data(ctx):
    ctx.run("dvc pull -r poke_store")


@task
def push_data(ctx):
    ctx.run("dvc push -r poke_store")


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


# TODO: Change the task so it works with uv manager
@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(
        f"python src/{PROJECT_NAME}/data.py --raw-data-path data/raw/dataset --output-folder data/processed",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(ctx: Context) -> None:
    """Train model with best config from sweep."""
    ctx.run(
        f"python src/{PROJECT_NAME}/train.py --num-classes 1000 --batch-size 32 --num-epochs 20 --lr 0.0014630344466593833 --wd 0.025329564397808917 --use-wandb --no-profiling --export-model --no-sweep",
        echo=True,
        pty=not WINDOWS,
    )


@task
def evaluate(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(f"python src/{PROJECT_NAME}/evaluate.py 31", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context) -> None:
    """Build docker images."""
    ctx.run(
        "docker build -t train:latest . -f dockerfiles/train.dockerfile",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        "docker build -t api:latest . -f dockerfiles/api.dockerfile",
        echo=True,
        pty=not WINDOWS,
    )


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run(
        "mkdocs build --config-file docs/mkdocs.yaml --site-dir build",
        echo=True,
        pty=not WINDOWS,
    )


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
