# ComfyUI wrapper for Kwai-Kolors

Rudimentary wrapper that runs Kwai-Kolors text2image pipeline using diffusers.

## Installation:

Clone this repository to 'ComfyUI/custom_nodes` folder.

Install the dependencies in requirements.txt, transformers version 4.38.0 minimum is required:

`pip install -r requirements.txt`

or if you use portable (run this in ComfyUI_windows_portable -folder):

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-KwaiKolorsWrapper\requirements.txt`


Models (fp16, 16.5GB) are automatically downloaded from https://huggingface.co/Kwai-Kolors/Kolors/tree/main

to `ComfyUI/models/diffusers/Kolors`

To run this, the text enconder is what takes most of the VRAM, but can be quantized to fit approximately these amounts:

| Model | Size | 
|--------|------| 
| fp16 | ~13 GB|
| quant8 | ~8 GB | 
| quant4 | ~4 GB |

After that, the sampling single image at 1024 can be expected to take similar amounts than SDXL. For VAE the base SDXL VAE is used.

![image](https://github.com/kijai/ComfyUI-KwaiKolorsWrapper/assets/40791699/ada4ac93-58ee-4957-96cd-2b327579d4f8)
