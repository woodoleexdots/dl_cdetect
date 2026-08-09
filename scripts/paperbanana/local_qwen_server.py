"""Local Qwen-Image generation server for PaperBanana.

Loads the 20B Qwen-Image pipeline once and serves POST /generate:
  {"prompt": str, "aspect_ratio": "16:9", "image_size": "2K"(ignored)}
  -> {"image_b64": <base64 PNG>}

Run:  python local_qwen_server.py   (port 8901)
"""

from __future__ import annotations

import base64
import io
import threading

import torch
from flask import Flask, jsonify, request

MODEL_DIR = "/home/wdlee/models/Qwen-Image"
PORT = 8901

ASPECT = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "21:9": (1664, 928),
}

app = Flask(__name__)
_pipe = None
_lock = threading.Lock()


def get_pipe():
    global _pipe
    if _pipe is None:
        from diffusers import DiffusionPipeline
        print("loading Qwen-Image ...", flush=True)
        p = DiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
        try:
            p.to("cuda")
            print("pipeline fully on GPU", flush=True)
        except torch.cuda.OutOfMemoryError:
            print("OOM on full load -> enabling CPU offload", flush=True)
            p.enable_model_cpu_offload()
        _pipe = p
        print("Qwen-Image ready", flush=True)
    return _pipe


@app.route("/health")
def health():
    return jsonify(ok=True, loaded=_pipe is not None)


@app.route("/generate", methods=["POST"])
def generate():
    d = request.get_json(force=True)
    prompt = d["prompt"]
    w, h = ASPECT.get(d.get("aspect_ratio", "16:9"), ASPECT["16:9"])
    steps = int(d.get("steps", 40))
    seed = int(d.get("seed", 0))
    with _lock:
        pipe = get_pipe()
        img = pipe(
            prompt=prompt,
            negative_prompt=" ",
            width=w,
            height=h,
            num_inference_steps=steps,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return jsonify(image_b64=base64.b64encode(buf.getvalue()).decode())


if __name__ == "__main__":
    get_pipe()  # warm load before serving
    app.run(host="127.0.0.1", port=PORT, threaded=True)
