%cd /kaggle/working/voxcpm_ft/voxcpm

!pip install -e .

!python -c "from huggingface_hub import snapshot_download; snapshot_download('JayLL13/VoxCPM-1.5-VN', local_dir='pretrained/VoxCPM-1.5-VN')"

!python /Users/macbook/Desktop/VoxCPM_Fine_tuning/voxcpm/scripts/test_voxcpm_lora_infer.py\
    --config_path "/voxcpm/config_lora.yaml" \
    --lora_ckpt "/voxcpm/checkpoints/step_0000800" \
    --text "Con biết không, dù ngoài kia thế giới có rộng lớn và khắc nghiệt đến đâu, thì nhà mình vẫn luôn là nơi bình yên nhất chờ đón con trở về. " \
    --prompt_audio "/VoxCPM_Fine_tuning/ref/ref_audio.wav" \
    --prompt_text "[Lo âu]: Nam ơi... Nắng xỉu, mà kiếm hổng ra chỗ để xe." \
    --output "/ket_qua_cloning.wav" \
    --cfg_value 3.0 \
    --inference_timesteps 30
