#!/usr/bin/env python3
import argparse
import os
import soundfile as sf
import torch

# ==============================================================================
# TỐI ƯU HÓA CHO RTX 5090 (Kiến trúc Ampere/Ada/Blackwell trở lên)
# 1. Bật TF32 (TensorFloat-32) để tăng tốc phép nhân ma trận trên Tensor Cores
# ==============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Ép backend soundfile để tránh lỗi IO
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig
from voxcpm.training.config import load_yaml_config

def parse_args():
    parser = argparse.ArgumentParser("VoxCPM LoRA Voice Cloning Inference (RTX 5090 Optimized)")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--lora_ckpt", type=str, required=True)
    parser.add_argument("--text", type=str, required=True, help="Nội dung cần nói")
    parser.add_argument("--prompt_audio", type=str, required=True, help="File âm thanh mẫu (WAV)")
    parser.add_argument("--prompt_text", type=str, required=True, help="Transcript của file mẫu")
    parser.add_argument("--output", type=str, default="cloning_result.wav")
    parser.add_argument("--cfg_value", type=float, default=3.0)
    parser.add_argument("--inference_timesteps", type=int, default=25)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load cấu hình
    cfg = load_yaml_config(args.config_path)
    pretrained_path = cfg["pretrained_path"]
    lora_cfg_dict = cfg.get("lora", {}) or {}
    lora_cfg = LoRAConfig(**lora_cfg_dict) if lora_cfg_dict else None

    # Khởi tạo Model
    print(f"[*] Loading model: {pretrained_path}")
    model = VoxCPM.from_pretrained(
        hf_model_id=pretrained_path,
        load_denoiser=False,
        optimize=True,
        lora_config=lora_cfg,
        lora_weights_path=args.lora_ckpt,
    )

    # ==============================================================================
    # 2. Chuyển model sang bfloat16. 
    # RTX 5090 có phần cứng hỗ trợ Native BF16 cực kỳ mạnh mẽ.
    # ==============================================================================
    model = model.to('cuda', dtype=torch.bfloat16)
    model.eval()

    # ==============================================================================
    # 3. Sử dụng torch.compile để tăng tốc đồ thị tính toán (Tuỳ chọn)
    # Lưu ý: Lần chạy ĐẦU TIÊN sẽ mất khoảng 1-2 phút để compile. 
    # Nhưng các file âm thanh thứ 2, thứ 3... sẽ được render siêu tốc.
    # Nếu bạn chỉ chạy 1 câu duy nhất rồi tắt app thì nên comment dòng này lại.
    # ==============================================================================
    # model = torch.compile(model, mode="reduce-overhead")

    print(f"[*] Đang thực hiện cloning tối ưu hóa trên RTX 5090...")
    
    # 4. Sử dụng context torch.inference_mode() và autocast (bfloat16)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        audio_np = model.generate(
            text=args.text,
            prompt_wav_path=args.prompt_audio,
            prompt_text=args.prompt_text,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
            denoise=False,
        )

    # Lưu kết quả
    sf.write(args.output, audio_np, model.tts_model.sample_rate)
    print(f"[OK] Đã lưu file tại: {args.output}")

if __name__ == "__main__":
    main()