import torch
import os
import gc

import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

from .kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from .kolors.models.modeling_chatglm import ChatGLMModel
from .kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler

from comfy.utils import ProgressBar

class DownloadAndLoadKolorsModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'Kwai-Kolors/Kolors',
                    ],
                    ),
            "precision": ([ 'fp16'],
                    {
                    "default": 'fp16'
                    }),
            },
        }

    RETURN_TYPES = ("KOLORSMODEL",)
    RETURN_NAMES = ("kolors_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "KwaiKolorsWrapper"

    def loadmodel(self, model, precision):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        pbar = ProgressBar(4)

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "diffusers", model_name)
      
        if not os.path.exists(model_path):
            print(f"Downloading Kolor model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            allow_patterns=['*fp16.safetensors*', '*.json', 'text_encoder/*', 'tokenizer/*'],
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
        pbar.update(1)
        print("Load VAE...")
        vae = AutoencoderKL.from_pretrained(model_path, subfolder='vae', revision=None, variant="fp16").to(dtype)

        pbar.update(1)

        scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        print("Load UNET...")
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder= 'unet', variant="fp16", revision=None, low_cpu_mem_usage=True).to(dtype).eval()
        print("Load TEXT_ENCODER...")
        pbar.update(1)

        text_encoder_path = os.path.join(model_path, "text_encoder")
        text_encoder = ChatGLMModel.from_pretrained(
            text_encoder_path,
            torch_dtype=dtype
            )
        tokenizer = ChatGLMTokenizer.from_pretrained(text_encoder_path)
        pbar.update(1)
        pipeline = StableDiffusionXLPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                force_zeros_for_empty_prompt=False
                )
        
        pipeline = pipeline.to(device)
        pipeline.enable_model_cpu_offload()
    
        kolors_model = {
            'pipeline': pipeline, 
            'dtype': dtype
            }

        return (kolors_model,)

    
class KolorsTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kolors_model": ("KOLORSMODEL", ),
                "prompt": ("STRING", {"multiline": True, "default": "",}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "",}),
            },
        }
    
    RETURN_TYPES = ("KOLORS_EMBEDS",)
    RETURN_NAMES =("kolors_embeds",)
    FUNCTION = "encode"
    CATEGORY = "KwaiKolorsWrapper"

    def encode(self, kolors_model, prompt, negative_prompt):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()

        num_images_per_prompt = 1
        do_classifier_free_guidance = True

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [kolors_model['pipeline'].tokenizer]
        text_encoders = [kolors_model['pipeline'].text_encoder]

        # textual inversion: procecss multi-vector tokens if necessary
        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            output = text_encoder(
                    input_ids=text_inputs['input_ids'] ,
                    attention_mask=text_inputs['attention_mask'],
                    position_ids=text_inputs['position_ids'],
                    output_hidden_states=True)
            
            prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone() # [batch_size, 77, 4096]
            text_proj = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            prompt_embeds_list.append(prompt_embeds)

        # prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        prompt_embeds = prompt_embeds_list[0]

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        # if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
        #     negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        #     negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        if do_classifier_free_guidance:
            uncond_tokens = []
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):            

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to('cuda')
                output = text_encoder(
                        input_ids=uncond_input['input_ids'] ,
                        attention_mask=uncond_input['attention_mask'],
                        position_ids=uncond_input['position_ids'],
                        output_hidden_states=True)
                negative_prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone() # [batch_size, 77, 4096]
                negative_text_proj = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]

                if do_classifier_free_guidance:
                    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                    seq_len = negative_prompt_embeds.shape[1]

                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(
                        batch_size * num_images_per_prompt, seq_len, -1
                    )

                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            # negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
            negative_prompt_embeds = negative_prompt_embeds_list[0]

        bs_embed = text_proj.shape[0]
        text_proj = text_proj.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        negative_text_proj = negative_text_proj.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        kolors_embeds = {
            'prompt_embeds': prompt_embeds,
            'negative_prompt_embeds': negative_prompt_embeds,
            'pooled_prompt_embeds': text_proj,
            'negative_pooled_prompt_embeds': negative_text_proj
        }
        
        return (kolors_embeds,)


class KolorsSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "kolors_model": ("KOLORSMODEL", ),
            "kolors_embeds": ("KOLORS_EMBEDS", ),
            #"latent": ("LATENT", ),
            "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "process"
    CATEGORY = "KwaiKolorsWrapper"

    def process(self, kolors_model, kolors_embeds, width, height, seed, steps, cfg):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        pipeline = kolors_model['pipeline']
        dtype = kolors_model['dtype']

        generator= torch.Generator(device).manual_seed(seed)

        image = pipeline(
            prompt=None,
            prompt_embeds = kolors_embeds['prompt_embeds'],
            pooled_prompt_embeds = kolors_embeds['pooled_prompt_embeds'],
            negative_prompt_embeds = kolors_embeds['negative_prompt_embeds'],
            negative_pooled_prompt_embeds = kolors_embeds['negative_pooled_prompt_embeds'],
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_images_per_prompt=1,
            generator= generator,
            output_type="pt",
            ).images[0]
        print(type(image))
        print(image.shape)

        tensor_out = image.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
        print(tensor_out.shape)
        print(tensor_out.min(), tensor_out.max())
        

        return (tensor_out,)   
     
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadKolorsModel": DownloadAndLoadKolorsModel,
    "KolorsSampler": KolorsSampler,
    "KolorsTextEncode": KolorsTextEncode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadKolorsModel": "(Down)load Kolors Model",
    "KolorsSampler": "Kolors Sampler",
    "KolorsTextEncode": "Kolors Text Encode"
}