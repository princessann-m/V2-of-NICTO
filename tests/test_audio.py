"""Tests for audio models."""

from __future__ import annotations

import numpy as np

import pytest

from mom.models.audio.config import AudioConfig
from mom.models.audio.stt import SpeechToText
from mom.models.audio.tts import TextToSpeech


def test_audio_config_defaults():
    cfg = AudioConfig()
    assert cfg.sample_rate == 16000
    assert cfg.n_mels == 80


def test_stt_transcribe():
    model = SpeechToText()
    audio = np.zeros(16000, dtype=np.float32)
    text = model.transcribe(audio)
    assert isinstance(text, str)


def test_stt_preprocess_shape():
    model = SpeechToText()
    audio = np.zeros(16000, dtype=np.float32)
    mel = model.preprocess(audio)
    assert mel.ndim == 3


def test_stt_marked_untrained():
    model = SpeechToText()
    assert model.trained is False


def test_tts_synthesize():
    model = TextToSpeech()
    audio = model.synthesize("hello")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32


def test_tts_marked_untrained():
    model = TextToSpeech()
    assert model.trained is False
