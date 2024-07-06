# ComfyUI wrapper for Kwai-Kolors

Rudimentary wrapper that runs Kwai-Kolors text2image pipeline using diffusers.

Models (fp16, 16.5GB) are automatically downloaded from https://huggingface.co/Kwai-Kolors/Kolors/tree/main

to `ComfyUI/models/diffusers/Kolors`

For now the text enconder seems to take ~13-15GB VRAM, while the inference after that only ~9GB.

![image](https://github.com/kijai/ComfyUI-KwaiKolorsWrapper/assets/40791699/7d9ef221-bb0e-4ec5-a9aa-9e46151118f0)

