import numpy as np
import torch
from types import SimpleNamespace

from app.model import EEGEncoder, MindToScriptModel


class DummyTokenizer:
    def batch_decode(self, sequences, skip_special_tokens=True):
        # return one decoded string per sequence
        batch = sequences.shape[0] if hasattr(sequences, "shape") else len(sequences)
        return ["decoded"] * batch


class DummyDecoder:
    def __init__(self, vocab_size: int = 50):
        self.vocab_size = vocab_size

    def generate(self, encoder_outputs=None, max_length=128, num_beams=1, return_dict_in_generate=True, output_scores=True):
        # encoder_outputs.last_hidden_state shape: (batch, seq_len, d_model)
        batch = encoder_outputs.last_hidden_state.shape[0]
        # produce sequences of length 5 with dummy token ids
        sequences = torch.randint(0, self.vocab_size, (batch, 5))
        # produce scores for 4 decode steps (list of tensors shaped (batch, vocab))
        scores = [torch.randn(batch, self.vocab_size) for _ in range(4)]
        return SimpleNamespace(sequences=sequences, scores=scores)


def test_encoder_forward_shape():
    batch = 2
    channels = 4
    samples = 200
    x = torch.randn(batch, channels, samples)
    enc = EEGEncoder(in_channels=channels, cnn_channels=16, lstm_hidden=32)
    out = enc(x)
    assert out.shape[0] == batch
    # seq_len equals samples (conv preserves length), hidden*2 equals lstm_hidden*2
    assert out.shape[1] == samples
    assert out.shape[2] == 32 * 2


def test_predict_returns_text_and_confidence():
    # prepare model with dummy tokenizer & decoder
    model = MindToScriptModel(device="cpu")
    model.tokenizer = DummyTokenizer()
    model.decoder = DummyDecoder()
    # set expected decoder hidden dim and ensure encoder/projection will be initialized accordingly
    model._d_model = 64

    # create fake EEG signals: channels x samples
    signals = np.random.randn(4, 256).astype(float)
    texts, confidences = model.predict(signals, max_length=10, num_beams=1)
    assert isinstance(texts, list)
    assert isinstance(confidences, list)
    assert len(texts) == len(confidences) == 1
    assert isinstance(texts[0], str)
    assert isinstance(confidences[0], float)

