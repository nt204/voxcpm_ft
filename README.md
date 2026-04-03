%cd /kaggle/working/voxcpm_ft/voxcpm

!pip install -e .

!python -c "from huggingface_hub import snapshot_download; snapshot_download('JayLL13/VoxCPM-1.5-VN', local_dir='pretrained/VoxCPM-1.5-VN')"

# thử với card thường
!python scripts/test_voxcpm_lora_infer.py \
--config_path "config_lora.yaml" \
--lora_ckpt "checkpoints/step_0000800" \
--text "Con gái nhỏ của bố, dù ngoài kia thế giới có bao la và nhiều giông bão đến đâu, thì vòng tay bố vẫn luôn là nơi an toàn nhất dành cho con. Cứ dũng cảm theo đuổi ước mơ của mình nhé. Mệt mỏi thì về, có bố đợi." \
--prompt_audio "/kaggle/working/voxcpm_ft/ref/ref_audio.wav" \
--prompt_text "[Lo âu]: Nam ơi... Nắng xỉu, mà kiếm hổng ra chỗ để xe." \
--output "/kaggle/working/ket_qua_cloning.wav" \
--cfg_value 3.0 \
--inference_timesteps 50

# thử với 5090
!python /scripts/test_voxcpm_5090.py\
--config_path "config_lora.yaml" \
--lora_ckpt "checkpoints/step_0000800" \
--text "Con gái nhỏ của bố, dù ngoài kia thế giới có bao la và nhiều giông bão đến đâu, thì vòng tay bố vẫn luôn là nơi an toàn nhất dành cho con. Cứ dũng cảm theo đuổi ước mơ của mình nhé. Mệt mỏi thì về, có bố đợi." \
--prompt_audio "/kaggle/working/voxcpm_ft/ref/ref_audio.wav" \
--prompt_text "[Lo âu]: Nam ơi... Nắng xỉu, mà kiếm hổng ra chỗ để xe." \
--output "/kaggle/working/ket_qua_cloning.wav" \
--cfg_value 3.0 \
--inference_timesteps 50
