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

Model folder structure needs to be the following:

```
PS C:\ComfyUI_windows_portable\ComfyUI\models\diffusers\Kolors> tree /F
│   model_index.json
│
├───scheduler
│       scheduler_config.json
│
├───text_encoder
│       config.json
│       configuration_chatglm.py
│       modeling_chatglm.py
│       pytorch_model-00001-of-00007.bin
│       pytorch_model-00002-of-00007.bin
│       pytorch_model-00003-of-00007.bin
│       pytorch_model-00004-of-00007.bin
│       pytorch_model-00005-of-00007.bin
│       pytorch_model-00006-of-00007.bin
│       pytorch_model-00007-of-00007.bin
│       pytorch_model.bin.index.json
│       quantization.py
│       tokenization_chatglm.py
│       tokenizer.model
│       tokenizer_config.json
│       vocab.txt
│
├───tokenizer
│       tokenization_chatglm.py
│       tokenizer.model
│       tokenizer_config.json
│       vocab.txt
│
├───unet
│       config.json
│       diffusion_pytorch_model.fp16.safetensors
│
└───vae
        config.json
        diffusion_pytorch_model.fp16.safetensors
```
To run this, the text enconder is what takes most of the VRAM, but can be quantized to fit approximately these amounts:

| Model | Size | 
|--------|------| 
| fp16 | ~13 GB|
| quant8 | ~8 GB | 
| quant4 | ~4 GB |

After that, the sampling single image at 1024 can be expected to take similar amounts than SDXL. For VAE the base SDXL VAE is used.

![image](https://github.com/kijai/ComfyUI-KwaiKolorsWrapper/assets/40791699/ada4ac93-58ee-4957-96cd-2b327579d4f8)

![image](https://github.com/kijai/ComfyUI-KwaiKolorsWrapper/assets/40791699/b6a17074-be09-4075-b66f-7857c871057a)

