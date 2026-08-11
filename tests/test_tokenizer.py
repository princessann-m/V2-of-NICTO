"""Tests for BPE tokenizer and vocabulary."""

from __future__ import annotations

import pytest

from mom.models.tokenizer import BPETokenizer, BPETrainer, Vocabulary


CORPUS = (
    "hello world hello hello world hello world hello world "
    "the quick brown fox jumps over the lazy dog "
    "hello world hello hello world hello world hello world "
)


class TestVocabulary:
    def test_special_tokens_exist(self):
        vocab = Vocabulary(vocab_size=100)
        assert vocab.pad_id >= 0
        assert vocab.bos_id >= 0
        assert vocab.eos_id >= 0
        assert vocab.unk_id >= 0

    def test_token_to_id_mapping(self):
        vocab = Vocabulary(vocab_size=100)
        vocab._add_token("hello")
        assert "hello" in vocab.token_to_id

    def test_id_to_token_mapping(self):
        vocab = Vocabulary(vocab_size=100)
        idx = vocab._add_token("world")
        assert vocab.id_to_token[idx] == "world"

    def test_save_and_load(self, tmp_path):
        vocab = Vocabulary(vocab_size=100)
        vocab._add_token("hello")
        vocab._add_token("world")
        path = str(tmp_path / "vocab.json")
        vocab.save(path)
        loaded = Vocabulary.load(path)
        assert "hello" in loaded.token_to_id
        assert "world" in loaded.token_to_id

    def test_len(self):
        vocab = Vocabulary(vocab_size=100)
        vocab._add_token("hello")
        vocab._add_token("world")
        assert len(vocab) >= 4


class TestBPETrainer:
    def test_train_produces_vocab(self):
        trainer = BPETrainer(vocab_size=50)
        vocab = trainer.train(CORPUS)
        assert len(vocab) > 0

    def test_merge_operations_recorded(self):
        trainer = BPETrainer(vocab_size=100)
        vocab = trainer.train(CORPUS)
        assert len(vocab.merges) > 0


class TestBPETokenizer:
    def test_encode_returns_list(self):
        tokenizer = BPETokenizer(vocab_size=100)
        ids = tokenizer.encode("hello world")
        assert isinstance(ids, list)

    def test_encode_contains_special_tokens(self):
        tokenizer = BPETokenizer(vocab_size=100, max_length=32)
        ids = tokenizer.encode("hello")
        assert tokenizer.vocab.bos_id in ids
        assert tokenizer.vocab.eos_id in ids

    def test_decode_returns_string(self):
        tokenizer = BPETokenizer(vocab_size=100)
        ids = tokenizer.encode("hello")
        text = tokenizer.decode(ids)
        assert isinstance(text, str)

    def test_train_updates_vocabulary(self):
        tokenizer = BPETokenizer(vocab_size=50)
        initial = len(tokenizer.vocab)
        tokenizer.train(CORPUS)
        assert len(tokenizer.vocab) > initial

    def test_encode_decode_roundtrip(self):
        tokenizer = BPETokenizer(vocab_size=200)
        tokenizer.train(CORPUS)
        text = "hello world"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert "hello" in decoded
        assert "world" in decoded

    def test_max_length_padding(self):
        tokenizer = BPETokenizer(vocab_size=100, max_length=16)
        ids = tokenizer.encode("hi")
        assert len(ids) == 16

    def test_save_and_load(self, tmp_path):
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(CORPUS)
        path = str(tmp_path / "tokenizer.json")
        tokenizer.save(path)
        loaded = BPETokenizer.load(path, vocab_size=100, max_length=16)
        assert len(loaded.vocab) == len(tokenizer.vocab)
