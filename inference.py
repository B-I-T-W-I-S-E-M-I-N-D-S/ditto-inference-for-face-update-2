import librosa
import math
import os
import numpy as np
import random
import torch
import pickle

from stream_pipeline_offline import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    SDK.setup(source_path, output_path, **setup_kwargs)

    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    online_mode = SDK.online_mode
    if online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    SDK.close()

    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)

    print(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Ditto Talking-Head Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Core paths ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_root", type=str,
        default="./checkpoints/ditto_trt_Ampere_Plus",
        help="path to TRT model data_root",
    )
    parser.add_argument(
        "--cfg_pkl", type=str,
        default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
        help="path to cfg_pkl",
    )
    parser.add_argument("--audio_path",  type=str, required=True,  help="path to input .wav")
    parser.add_argument("--source_path", type=str, required=True,  help="path to input image/video")
    parser.add_argument("--output_path", type=str, required=True,  help="path to output .mp4")

    # ── Canonical face normalization ──────────────────────────────────────────
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Enable canonical face normalization (template geometry override).",
    )
    parser.add_argument(
        "--template_path", type=str, default="face.jpg",
        help="Path to the canonical template face image.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7,
        help="Geometry blend strength: 0=pure user, 1=pure template.",
    )
    parser.add_argument(
        "--mouth_alpha", type=float, default=0.3,
        help="Blend strength for mouth/lip keypoints (lower = better lip-sync).",
    )
    parser.add_argument(
        "--clamp_val", type=float, default=0.3,
        help="Clamp absolute keypoint delta to this value (prevents extreme warping). "
             "Set to 0 to disable clamping.",
    )
    parser.add_argument(
        "--no_preprocess_face",
        action="store_true",
        help="Disable face parsing/masking (skip hair/ear removal).",
    )
    parser.add_argument(
        "--fill_method", type=str, default="mean",
        choices=["mean", "blur", "black"],
        help="Background fill method after face masking.",
    )
    parser.add_argument(
        "--bisenet_path", type=str,
        default="checkpoints/bisenet/79999_iter.pth",
        help="Path to BiSeNet weights (.pth). Falls back to heuristic if absent.",
    )
    parser.add_argument(
        "--bisenet_device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for BiSeNet inference.",
    )

    args = parser.parse_args()

    # ── Build canonical_cfg ───────────────────────────────────────────────────
    canonical_cfg: dict = {"enabled": False}
    if args.canonical:
        canonical_cfg = {
            "enabled":          True,
            "template_path":    args.template_path,
            "alpha":            args.alpha,
            "mouth_alpha":      args.mouth_alpha,
            "clamp_val":        args.clamp_val if args.clamp_val > 0 else None,
            "preprocess_face":  not args.no_preprocess_face,
            "fill_method":      args.fill_method,
            "bisenet_mode":     "bisenet",   # auto-falls back to heuristic
            "bisenet_path":     args.bisenet_path,
            "bisenet_device":   args.bisenet_device,
        }
        print("[inference] Canonical normalization settings:")
        for k, v in canonical_cfg.items():
            print(f"  {k}: {v}")

    # ── Initialise SDK ────────────────────────────────────────────────────────
    SDK = StreamSDK(args.cfg_pkl, args.data_root, canonical_cfg=canonical_cfg)

    # ── Run ───────────────────────────────────────────────────────────────────
    # seed_everything(1024)
    run(SDK, args.audio_path, args.source_path, args.output_path)

