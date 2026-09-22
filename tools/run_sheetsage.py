"""Transcribe an audio file with SheetSage2 from the shared HF cache (offline, pinned revision).

usage: python run_sheetsage.py AUDIO OUT_DIR
"""
import sys
from transformers import AutoModel

REV = "7cb7b16233d2f5ccb4d3295c2b501fdc31bc0384"   # the cached snapshot that holds model.safetensors
audio, out = sys.argv[1], sys.argv[2]

model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True,
                                  revision=REV, code_revision=REV).eval().to("cuda")
r = model.transcribe(audio, output_dir=out, dtype="bf16", preset="default")
print({k: r.get(k) for k in ("duration_seconds", "elapsed_seconds", "peak_gpu_mib", "melody_notes",
                             "vocal_notes", "instrumental_notes", "abc_measures", "warnings", "diagnostics")})
