#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import soundfile as sf

# Ép backend soundfile để tránh lỗi trên Kaggle
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig
from voxcpm.training.config import load_yaml_config

def parse_args():
    parser = argparse.ArgumentParser("VoxCPM LoRA Voice Cloning Inference")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--lora_ckpt", type=str, required=True)
    parser.add_argument("--text", type=str, required=True, help="Nội dung cần nói")
    parser.add_argument("--prompt_audio", type=str, required=True, help="File âm thanh mẫu (WAV)")
    parser.add_argument("--prompt_text", type=str, required=True, help="Transcript của file mẫu")
    parser.add_argument("--output", type=str, default="cloning_result.wav")
    parser.add_argument("--cfg_value", type=float, default=3.5)
    parser.add_argument("--inference_timesteps", type=int, default=25) # Giảm xuống 25 cho nhanh trên T4
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load cấu hình LoRA từ file YAML
    cfg = load_yaml_config(args.config_path)
    pretrained_path = cfg["pretrained_path"]
    lora_cfg_dict = cfg.get("lora", {}) or {}
    lora_cfg = LoRAConfig(**lora_cfg_dict) if lora_cfg_dict else None

    # 2. Khởi tạo Model
    print(f"[*] Loading model: {pretrained_path}")
    model = VoxCPM.from_pretrained(
        hf_model_id=pretrained_path,
        load_denoiser=False,
        optimize=True,
        lora_config=lora_cfg,
        lora_weights_path=args.lora_ckpt,
    )

    # 3. Tiến hành Voice Cloning
    print(f"[*] Đang thực hiện cloning...")
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=args.prompt_audio,
        prompt_text=args.prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        denoise=False,
    )

    # 4. Lưu kết quả
    sf.write(args.output, audio_np, model.tts_model.sample_rate)
    print(f"[OK] Đã lưu file tại: {args.output}")

if __name__ == "__main__":
    main()