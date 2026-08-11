"""Tests for training recipes."""

from __future__ import annotations

import os

import pytest
import yaml

from mom.training.recipes import Recipe
from mom.training.recipes.mamba_small import MambaSmallRecipe
from mom.training.recipes.mamba_medium import MambaMediumRecipe
from mom.training.recipes.mamba_large import MambaLargeRecipe
from mom.training.recipes.top_model import TopModelRecipe
from mom.training.recipes.verifier import VerifierRecipe
from mom.training.recipes.judge import JudgeRecipe
from mom.training.recipes.gan import GanRecipe
from mom.training.recipes.director import DirectorRecipe


RECIPE_FILES = [
    "mom/training/recipes/mamba_small.yaml",
    "mom/training/recipes/mamba_medium.yaml",
    "mom/training/recipes/mamba_large.yaml",
    "mom/training/recipes/top_model.yaml",
    "mom/training/recipes/verifier.yaml",
    "mom/training/recipes/judge.yaml",
    "mom/training/recipes/gan.yaml",
    "mom/training/recipes/director.yaml",
]


def test_recipe_files_exist():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in RECIPE_FILES:
        path = os.path.join(base, rel)
        assert os.path.exists(path), f"Missing recipe: {path}"


def test_load_recipe_from_file(tmp_path):
    recipe_path = tmp_path / "test_recipe.yaml"
    with open(recipe_path, "w") as f:
        yaml.safe_dump({"name": "test", "d_model": 768, "epochs": 5}, f)
    recipe = Recipe.from_file(str(recipe_path))
    assert recipe.name == "test"
    assert recipe.config["d_model"] == 768
    assert recipe.config["epochs"] == 5


def test_recipe_save_roundtrip(tmp_path):
    recipe = Recipe(name="roundtrip", config={"epochs": 3, "batch_size": 8})
    out = tmp_path / "out.yaml"
    recipe.save(str(out))
    reloaded = Recipe.from_file(str(out))
    assert reloaded.name == "roundtrip"
    assert reloaded.config["epochs"] == 3


def test_mamba_small_config():
    cfg = MambaSmallRecipe()
    assert cfg.config["d_model"] == 768
    assert cfg.config["n_layers"] == 24
    assert cfg.config["training"]["batch_size"] == 4


def test_mamba_medium_config():
    cfg = MambaMediumRecipe()
    assert cfg.config["d_model"] == 1024
    assert cfg.config["n_layers"] == 36
    assert cfg.config["training"]["distributed"] is True


def test_mamba_large_config():
    cfg = MambaLargeRecipe()
    assert cfg.config["d_model"] == 1536
    assert cfg.config["n_layers"] == 48


def test_top_model_config():
    cfg = TopModelRecipe()
    assert "encoder" in cfg.config
    assert "decoder" in cfg.config
    assert cfg.config["training"]["epochs"] == 5


def test_verifier_config():
    cfg = VerifierRecipe()
    assert cfg.config["hidden_size"] == 256
    assert cfg.config["num_layers"] == 4


def test_judge_config():
    cfg = JudgeRecipe()
    assert cfg.config["hidden_size"] == 256
    assert cfg.config["num_layers"] == 4


def test_gan_config():
    cfg = GanRecipe()
    assert "generator" in cfg.config
    assert "discriminator" in cfg.config
    assert cfg.config["training"]["learning_rate_g"] == 2e-4


def test_director_config():
    cfg = DirectorRecipe()
    assert cfg.config["hidden_size"] == 768
    assert cfg.config["num_layers"] == 12
    assert cfg.config["training"]["epochs"] == 5
