"""Real test suite for MoM framework."""

from __future__ import annotations

import time
import torch
import pytest

from mom.config import MoMConfig, LLMConfig, load_config
from mom.models.top_model import TopModel
from mom.models.model_registry import ModelRegistry
from mom.models.llm_client import LLMClient
from mom.tools.calculator import Calculator
from mom.tools.coding_studio import CodingStudio
from mom.tools.simulator import Simulator
from mom.verification.hallucination import HallucinationChecker
from mom.experts.math_expert import MathExpert
from mom.experts.coding_expert import CodingExpert
from mom.experts.reasoning_expert import ReasoningExpert
from mom.experts.science_expert import ScienceExpert
from mom.experts.vision_expert import VisionExpert
from mom.experts.mme import MixedMambaExpertSystem
from mom.core.orchestrator import Orchestrator
from mom.judges.judge import Judge
from mom.models.mamba import MambaModel, MambaConfig, MambaExpert
from mom.models.encoder_decoder import EncoderDecoderModel, TopModel as RealTopModel
from mom.models.encoder_decoder.config import EncoderDecoderConfig
from mom.generation.gan import Generator, Discriminator, GANTrainer, GANConfig
from mom.tools.simulation import SimulationEngine
from mom.tools.simulation.physics import PendulumSimulator
from mom.tools.virtual_lab import VirtualLab
from mom.tools.virtual_lab.experiments import Experiment
from mom.core.routing.master_router import MasterRouter
from mom.core.routing.expert_router import ExpertRouter
from mom.models.verifier import VerifierModel, VerifierConfig
from mom.models.judge import JudgeModel, JudgeConfig
from mom.mme.system import MMESystem
from mom.mme.orchestrator import MMEOrchestrator
from mom.generation.video import VideoPipeline, VideoPipelineConfig
from mom.generation.image import ImagePipeline


# --- Config ---

def test_load_config_defaults():
    cfg = load_config()
    assert cfg.llm.provider == "heuristic"
    assert cfg.global_deadline == 10.0
    assert cfg.max_retries == 1


# --- Mamba ---

def test_mamba_forward():
    cfg = MambaConfig(vocab_size=100, d_model=32, n_layers=2, d_state=16, d_conv=4, expand=2)
    model = MambaModel(cfg)
    x = torch.randint(0, 100, (2, 16))
    logits, loss = model(x)
    assert logits.shape == (2, 16, 100)


def test_mamba_expert():
    cfg = MambaConfig(vocab_size=100, d_model=32, n_layers=2, d_state=16, d_conv=4, expand=2)
    expert = MambaExpert(config=cfg, role="math_expert", capabilities=["algebra"])
    assert expert.status == "UNTRAINED"
    result = expert.compute(torch.randint(0, 100, (1, 8)))
    assert "output_ids" in result


# --- Encoder-Decoder ---

def test_encoder_decoder_forward():
    cfg = EncoderDecoderConfig(vocab_size=100, hidden_size=32, num_attention_heads=2, num_encoder_layers=2, num_decoder_layers=2, feed_forward_size=64)
    model = EncoderDecoderModel(cfg)
    src = torch.randint(0, 100, (2, 8))
    tgt = torch.randint(0, 100, (2, 8))
    out = model(src, decoder_input_ids=tgt)
    assert "logits" in out


# --- GAN ---

def test_gan_forward():
    G = Generator(latent_dim=16, image_channels=1, image_size=8)
    D = Discriminator(image_channels=1, image_size=8)
    z = torch.randn(2, 16)
    fake = G(z)
    assert fake.shape == (2, 1, 8, 8)
    pred = D(fake)
    assert pred.shape == (2, 1)


# --- Simulation ---

def test_simulation_pendulum():
    engine = SimulationEngine()
    sim = PendulumSimulator()
    result = sim.run(engine, {"length": 1.0, "gravity": 9.81, "damping": 0.1, "theta0": 0.5, "omega0": 0.0, "t_span": (0, 2)})
    assert result.label == "SIMULATION RESULT"
    assert hasattr(result, "time")
    assert hasattr(result, "state")


def test_virtual_lab_experiment():
    lab = VirtualLab()
    exp = lab.create_experiment(
        question="Does pendulum period depend on length?",
        hypothesis="Longer pendulum has longer period",
        design="Measure period for different lengths",
    )
    def run_fn():
        return {"readings": [{"instrument": "stopwatch", "value": 2.01}]}
    result = lab.run(exp.experiment_id, run_fn)
    assert result.label == "VIRTUAL EXPERIMENT RESULT"


# --- Router ---

def test_master_router():
    reg = ModelRegistry()
    router = MasterRouter(reg)
    plan = router.select_mme(router.analyze_task("Calculate 2 + 2"))
    assert "selected_experts" in plan


def test_expert_router():
    reg = ModelRegistry()
    router = ExpertRouter(reg, k=2)
    result = router.route({"task_type": "math"})
    assert len(result.selected_ids) <= 2


# --- Verifier/Judge ---

def test_verifier_forward():
    cfg = VerifierConfig(vocab_size=100, hidden_size=32, num_layers=2, num_heads=2, ff_size=64)
    model = VerifierModel(cfg)
    x = torch.randint(0, 100, (2, 8))
    out = model(x)
    assert "score" in out


def test_judge_forward():
    cfg = JudgeConfig(vocab_size=100, hidden_size=32, num_layers=2, num_heads=2, ff_size=64)
    model = JudgeModel(cfg, name="test_judge")
    x = torch.randint(0, 100, (2, 8))
    out = model(x)
    assert "score" in out


# --- MME ---

def test_mme_system():
    reg = ModelRegistry()
    cfg = MambaConfig(vocab_size=100, d_model=32, n_layers=2, d_state=16, d_conv=4, expand=2)
    mme = MMESystem({"selected_experts": [{"name": "reasoning_expert", "domain": "reasoning"}]}, registry=reg)
    result = mme.process({"original": "test task"}, deadline=time.time() + 5)
    assert "answer" in result


def test_mme_orchestrator():
    reg = ModelRegistry()
    orch = MMEOrchestrator(reg)
    candidates = orch.run({"original": "test"}, deadline=time.time() + 5)
    assert len(candidates) == 3


# --- Orchestrator end-to-end ---

def test_orchestrator_math():
    orch = Orchestrator(MoMConfig(global_deadline=5.0))
    res = orch.handle_request("Calculate 2 + 2", request_id="test_math")
    assert isinstance(res, dict)
    assert "answer" in res


def test_orchestrator_coding():
    orch = Orchestrator(MoMConfig(global_deadline=5.0))
    res = orch.handle_request("Write a python function to add two numbers", request_id="test_code")
    assert isinstance(res, dict)


# --- Video/Image Pipeline ---

def test_video_pipeline():
    pipe = VideoPipeline(VideoPipelineConfig())
    result = pipe.generate("A story about AI")
    assert "status" in result


def test_image_pipeline():
    pipe = ImagePipeline()
    result = pipe.generate({"prompt": "a cat", "modality": "image"})
    assert "status" in result
