import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from types import SimpleNamespace


class EEGEncoder(nn.Module):
    """
    Simple 1D-CNN + BiLSTM encoder for EEG signals.
    Input: (batch, channels, samples)
    Output: (batch, seq_len, hidden*2)
    """
    def __init__(self, in_channels: int, cnn_channels: int = 64, lstm_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
        )
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, samples) -> apply conv over time using channels as in_channels
        # We need shape (batch, in_channels, seq_len) where seq_len = samples
        out = self.conv(x)  # (batch, cnn_channels, seq_len)
        out = out.permute(0, 2, 1)  # (batch, seq_len, cnn_channels)
        out, _ = self.lstm(out)  # (batch, seq_len, hidden*2)
        out = self.dropout(out)
        return out


class MindToScriptModel:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = None
        self.decoder = None
        self.encoder: Optional[EEGEncoder] = None
        self.projection: Optional[nn.Linear] = None
        self.hf_model_name = None

    def load(self, model_dir: Optional[str] = None, hf_model_name: Optional[str] = "facebook/bart-base"):
        """
        Load the HuggingFace decoder (BART) and prepare encoder/projection.
        If model_dir contains a HF-style checkpoint, load from there. Otherwise, download hf_model_name.
        """
        self.hf_model_name = hf_model_name
        # load tokenizer and decoder (import transformers lazily so tests/CI can skip installing transformers)
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except Exception as e:
            raise RuntimeError("transformers library is required to load HF decoder. Install 'transformers' or provide a local decoder.") from e
        source = model_dir if model_dir and os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")) else hf_model_name
        print("Loading decoder/tokenizer from", source)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.decoder = AutoModelForSeq2SeqLM.from_pretrained(source).to(self.device)
        self.decoder.eval()

        # projection will be created lazily once we know decoder hidden size and encoder output dim
        self._d_model = getattr(self.decoder.config, "d_model", 768)
        # Optionally, try to load bridge weights if present
        if model_dir:
            # look for bridge.pt or encoder.pt
            bridge_ckpt = None
            for fname in ("bridge.pt", "encoder.pt", "bridge.pth", "encoder.pth"):
                p = os.path.join(model_dir, fname)
                if os.path.exists(p):
                    bridge_ckpt = p
                    break
            if bridge_ckpt:
                print("Found bridge checkpoint:", bridge_ckpt)
                # we'll load into encoder/projection after instantiation
                self._bridge_ckpt = bridge_ckpt
                # attempt to read saved config to initialize encoder with same dims
                try:
                    ck = torch.load(self._bridge_ckpt, map_location="cpu")
                    cfg = ck.get("config", {})
                    self._bridge_config = cfg
                except Exception:
                    self._bridge_config = {}
            else:
                self._bridge_ckpt = None
        else:
            self._bridge_ckpt = None

    def _ensure_encoder(self, in_channels: int):
        if self.encoder is None or self.encoder.conv[0].in_channels != in_channels:
            # consult bridge config if available
            cfg = getattr(self, "_bridge_config", {}) or {}
            cnn_channels = int(cfg.get("cnn_channels", 64))
            lstm_hidden = int(cfg.get("lstm_hidden", 128))
            chosen_in_ch = int(cfg.get("in_channels", in_channels))
            print(f"Initializing encoder for in_channels={chosen_in_ch}, cnn_channels={cnn_channels}, lstm_hidden={lstm_hidden}")
            self.encoder = EEGEncoder(in_channels=chosen_in_ch, cnn_channels=cnn_channels, lstm_hidden=lstm_hidden).to(self.device)
            # projection: map encoder hidden * 2 (from BiLSTM) to decoder d_model
            sample_hidden_dim = self.encoder.lstm.hidden_size * 2
            self.projection = nn.Linear(sample_hidden_dim, self._d_model).to(self.device)
            # If bridge checkpoint exists, try to load its state dict
            if getattr(self, "_bridge_ckpt", None):
                try:
                    sd = torch.load(self._bridge_ckpt, map_location=self.device)
                    if "encoder" in sd or "encoder_state_dict" in sd:
                        enc_sd = sd.get("encoder", sd.get("encoder_state_dict"))
                        try:
                            self.encoder.load_state_dict(enc_sd)
                        except Exception as e:
                            print("Warning: failed to fully load encoder state_dict:", e)
                    if "projection" in sd or "projection_state_dict" in sd:
                        proj_sd = sd.get("projection", sd.get("projection_state_dict"))
                        try:
                            self.projection.load_state_dict(proj_sd)
                        except Exception as e:
                            print("Warning: failed to fully load projection state_dict:", e)
                    print("Loaded bridge weights into encoder/projection.")
                except Exception as e:
                    print("Failed to load bridge ckpt:", e)

    def _preprocess(self, signals: np.ndarray) -> torch.Tensor:
        """
        signals: numpy array of shape (channels, samples) or (batch, channels, samples)
        returns torch tensor (batch, channels, samples) on device
        """
        arr = np.asarray(signals)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)  # batch dim
        # arr shape (batch, channels, samples)
        # z-score per channel
        mean = arr.mean(axis=2, keepdims=True)
        std = arr.std(axis=2, keepdims=True) + 1e-6
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).float().to(self.device)
        return tensor

    def compute_sqi(self, signals: np.ndarray) -> Tuple[float, dict]:
        """
        Compute a simple Signal Quality Index (SQI).
        signals: numpy array (channels, samples) or (batch, channels, samples)
        Returns (sqi_score in [0,1], details)
        Heuristics used:
         - flatline_ratio: fraction of channels with very low std
         - high_artifact_ratio: fraction of samples exceeding 6*std (spikes)
         - amplitude_range: median peak-to-peak normalized
        """
        arr = np.asarray(signals)
        if arr.ndim == 3:
            arr = arr[0]
        ch_std = arr.std(axis=1)
        ch_mean = arr.mean(axis=1)
        # flatline channels
        flatline_thresh = 1e-6
        flatline_ratio = float((ch_std < flatline_thresh).sum()) / max(1, arr.shape[0])
        # spikes
        z = (arr - ch_mean[:, None]) / (ch_std[:, None] + 1e-9)
        spike_mask = np.abs(z) > 6.0
        high_artifact_ratio = float(spike_mask.sum()) / (arr.size + 1e-9)
        # amplitude range normalized by per-channel std
        ptp = arr.ptp(axis=1)
        amp_norm = np.median(ptp / (ch_std + 1e-9))
        # fraction of channels with peak-to-peak amplitude exceeding ±100µV (common rejection threshold)
        bad_amp_fraction = float((ptp > 100.0).sum()) / max(1, arr.shape[0])
        # build a simple score: lower is worse; map to 0..1
        score = 1.0
        score -= 0.5 * flatline_ratio
        score -= 0.4 * min(1.0, high_artifact_ratio * 100.0)  # scale spikes influence
        # penalize extreme amplitude ranges
        if amp_norm > 50:
            score -= 0.2
        score = max(0.0, min(1.0, score))
        details = {
            "flatline_ratio": flatline_ratio,
            "high_artifact_ratio": high_artifact_ratio,
            "amp_norm_median": float(amp_norm),
            "bad_amp_fraction": float(bad_amp_fraction),
        }
        return score, details

    def predict(self, signals: np.ndarray, sfreq: Optional[float] = None, max_length: int = 128, num_beams: int = 1) -> Tuple[List[str], List[float]]:
        """
        signals: numpy array (channels, samples) or (batch, channels, samples)
        returns: (texts, confidences)
        """
        x = self._preprocess(signals)  # (batch, channels, samples)
        batch_size, in_channels, samples = x.shape
        self._ensure_encoder(in_channels)
        # use inference_mode to reduce graph/tracking overhead
        with torch.inference_mode():
            enc_out = self.encoder(x)  # (batch, seq_len, hidden*2)
            # project to decoder d_model
            proj = self.projection(enc_out)  # (batch, seq_len, d_model)
            # prepare encoder_outputs in the format expected by HF generate.
            # Prefer BaseModelOutput when transformers is available; fallback to dict.
            try:
                from transformers.modeling_outputs import BaseModelOutput
                encoder_outputs = BaseModelOutput(last_hidden_state=proj)
            except Exception:
                encoder_outputs = {"last_hidden_state": proj}
            # generate with scores for confidence estimation
            gen = self.decoder.generate(encoder_outputs=encoder_outputs, max_length=max_length, num_beams=num_beams, return_dict_in_generate=True, output_scores=True)
            sequences = gen.sequences  # (batch, seq_len_out)
            # compute simple confidence from per-step scores if available
            confidences = [0.0] * batch_size
            try:
                scores = gen.scores  # list(len=T) of tensors (batch, vocab)
                # for each step compute softmax max prob
                step_probs = []
                for s in scores:
                    probs = torch.softmax(s, dim=-1)  # (batch, vocab)
                    maxp, _ = probs.max(dim=-1)  # (batch,)
                    step_probs.append(maxp)
                # stack (steps, batch) -> (batch, steps)
                sp = torch.stack(step_probs, dim=1)  # (batch, steps)
                # mean across steps
                meanp = sp.mean(dim=1).tolist()
                confidences = meanp
            except Exception:
                confidences = [0.0] * batch_size
            texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        # optional: clear CUDA cache to reduce OOM risk in long-running servers
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return texts, confidences

