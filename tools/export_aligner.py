"""One-off: export the forced-alignment model to ONNX so sidecut can run it with
onnxruntime alone (no torch at run time).

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers onnx onnxruntime
    python tools/export_aligner.py OUTDIR

Writes OUTDIR/wav2vec2-base-960h.onnx (float32) and .int8.onnx (dynamically
quantised), plus labels.json with the model's character vocabulary, and checks
both against the torch model on a synthetic input.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL = "facebook/wav2vec2-base-960h"


class Wrapped(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio):
        return torch.log_softmax(self.model(audio).logits, dim=-1)


def main(outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    processor = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
    vocab = processor.tokenizer.get_vocab()
    labels = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    (out / "labels.json").write_text(json.dumps({
        "model": MODEL, "labels": labels, "blank": labels.index("<pad>"), "word_separator": "|",
        "sample_rate": 16000, "frame_ms": 20, "normalize": True,
    }, indent=1))

    wrapped = Wrapped(model)
    example = torch.randn(1, 16000 * 3)
    fp32 = out / "wav2vec2-base-960h.onnx"
    torch.onnx.export(
        wrapped, (example,), str(fp32), input_names=["audio"], output_names=["log_probs"],
        dynamic_axes={"audio": {0: "batch", 1: "samples"}, "log_probs": {0: "batch", 1: "frames"}},
        opset_version=17, dynamo=False,
    )
    with torch.no_grad():
        reference = wrapped(example).numpy()

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    session = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    got = session.run(None, {"audio": example.numpy()})[0]
    print(f"fp32: shape {got.shape}, max |diff| vs torch {np.abs(got - reference).max():.5f}")

    int8 = out / "wav2vec2-base-960h.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    session = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    got8 = session.run(None, {"audio": example.numpy()})[0]
    agree = (got8.argmax(-1) == reference.argmax(-1)).mean()
    print(f"int8: max |diff| {np.abs(got8 - reference).max():.3f}, argmax agreement {agree:.3f}")
    for path in (fp32, int8):
        print(f"{path.name}: {path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/aligner")
