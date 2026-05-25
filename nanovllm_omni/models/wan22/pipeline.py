import time
from pathlib import Path

import torch
from PIL import Image

from nanovllm_omni.cache import PromptCache
from nanovllm_omni.config import EngineConfig
from nanovllm_omni.models.interface import SupportsStepExecution
from nanovllm_omni.outputs import OmniOutput
from nanovllm_omni.utils import resize_with_aspect
from nanovllm_omni.worker.utils import RunnerState


class Wan22I2VPipeline(SupportsStepExecution):
    supports_step_execution = True

    def __init__(self, config: EngineConfig):
        from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline

        self.config = config
        self.device = torch.device(config.device)
        self.prompt_cache = PromptCache(config.prompt_cache_size)
        self._active_module_name: str | None = None
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            config.model,
            vae=AutoencoderKLWan.from_pretrained(
                config.model,
                subfolder="vae",
                torch_dtype=config.vae_dtype,
                local_files_only=config.local_files_only,
            ),
            torch_dtype=config.dtype,
            local_files_only=config.local_files_only,
        )
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        if config.use_cpu_offload:
            self.pipe.to("cpu")
        else:
            self.pipe.to(self.device)

    def _move_module(self, module_name: str, device: torch.device | str) -> None:
        module = getattr(self.pipe, module_name, None)
        if module is None:
            return
        module.to(device)

    def _manual_offload(self, keep: str | None = None) -> None:
        if not self.config.use_cpu_offload:
            return
        for module_name in ("text_encoder", "transformer", "transformer_2", "vae"):
            if module_name != keep:
                self._move_module(module_name, "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _set_active_module(self, module_name: str | None) -> None:
        if not self.config.use_cpu_offload:
            return
        if module_name == self._active_module_name:
            return
        if self._active_module_name is not None:
            self._move_module(self._active_module_name, "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if module_name is not None:
            self._move_module(module_name, self.device)
        self._active_module_name = module_name

    def _activate_module(self, module_name: str) -> None:
        if not self.config.use_cpu_offload:
            return
        self._manual_offload(keep=module_name)
        self._move_module(module_name, self.device)

    def _load_image(self, image: str | Path | Image.Image, height: int, width: int) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        return resize_with_aspect(image, height, width)

    def _encode_prompt_pair(self, state: RunnerState, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
        cache_key = (state.prompt, state.negative_prompt)
        cached = self.prompt_cache.get(cache_key)
        if cached is not None:
            prompt_embeds, negative_prompt_embeds = cached
            return prompt_embeds.to(device), None if negative_prompt_embeds is None else negative_prompt_embeds.to(device)
        self._set_active_module("text_encoder")
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=state.prompt,
            negative_prompt=state.negative_prompt,
            do_classifier_free_guidance=state.sampling.guidance_scale > 1.0,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
        self.prompt_cache.put(cache_key, (prompt_embeds, negative_prompt_embeds))
        return prompt_embeds, negative_prompt_embeds

    def prepare_encode(self, state: RunnerState) -> RunnerState:
        t0 = time.perf_counter()
        device = self.device
        image = self._load_image(state.image, state.sampling.height, state.sampling.width)
        state.extra["image"] = image
        state.extra["stage_durations"] = {"load_image": time.perf_counter() - t0}

        t1 = time.perf_counter()
        prompt_embeds, negative_prompt_embeds = self._encode_prompt_pair(state, device)
        state.prompt_embeds = prompt_embeds.to(self.pipe.transformer.dtype)
        state.negative_prompt_embeds = (
            None if negative_prompt_embeds is None else negative_prompt_embeds.to(self.pipe.transformer.dtype)
        )
        state.extra["stage_durations"]["encode_prompt"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        state.scheduler = self.pipe.scheduler.from_config(self.pipe.scheduler.config, flow_shift=state.sampling.flow_shift)
        state.scheduler.set_timesteps(state.sampling.num_inference_steps, device=device)
        state.timesteps = state.scheduler.timesteps

        self._set_active_module("vae")
        image_tensor = self.pipe.video_processor.preprocess(image, height=image.height, width=image.width).to(
            device, dtype=torch.float32
        )
        generator = torch.Generator(device=device).manual_seed(state.sampling.seed)
        latents, condition, first_frame_mask = self.pipe.prepare_latents(
            image_tensor,
            batch_size=1,
            num_channels_latents=self.pipe.vae.config.z_dim,
            height=image_tensor.shape[-2],
            width=image_tensor.shape[-1],
            num_frames=state.sampling.num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=None,
            last_image=None,
        )
        state.latents = latents
        state.extra["condition"] = condition
        state.extra["first_frame_mask"] = first_frame_mask
        state.extra["stage_durations"]["prepare_latents"] = time.perf_counter() - t2
        return state

    def denoise_step(self, state: RunnerState) -> torch.Tensor | None:
        timestep = state.current_timestep
        if timestep is None:
            return None
        self._set_active_module("transformer")
        condition = state.extra["condition"]
        first_frame_mask = state.extra["first_frame_mask"]
        transformer_dtype = self.pipe.transformer.dtype
        latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * state.latents
        latent_model_input = latent_model_input.to(transformer_dtype)
        timestep_tensor = (first_frame_mask[0][0][:, ::2, ::2] * timestep).flatten().unsqueeze(0)
        with self.pipe.transformer.cache_context("cond"):
            noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_tensor,
                encoder_hidden_states=state.prompt_embeds,
                encoder_hidden_states_image=None,
                attention_kwargs=None,
                return_dict=False,
            )[0]
        if state.negative_prompt_embeds is not None:
            with self.pipe.transformer.cache_context("uncond"):
                noise_uncond = self.pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_tensor,
                    encoder_hidden_states=state.negative_prompt_embeds,
                    encoder_hidden_states_image=None,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            noise_pred = noise_uncond + state.sampling.guidance_scale * (noise_pred - noise_uncond)
        state.extra["noise_pred"] = noise_pred
        return noise_pred

    def step_scheduler(self, state: RunnerState, noise_pred: torch.Tensor | None = None) -> None:
        timestep = state.current_timestep
        if timestep is None:
            return
        if noise_pred is None:
            noise_pred = state.extra.pop("noise_pred")
        else:
            state.extra.pop("noise_pred", None)
        state.latents = state.scheduler.step(noise_pred, timestep, state.latents, return_dict=False)[0]
        state.step_index += 1

    def post_decode(self, state: RunnerState) -> OmniOutput:
        t0 = time.perf_counter()
        self._set_active_module("vae")
        condition = state.extra["condition"]
        first_frame_mask = state.extra["first_frame_mask"]
        latents = (1 - first_frame_mask) * condition + first_frame_mask * state.latents
        latents = latents.to(self.pipe.vae.dtype)
        latents_mean = (
            torch.tensor(self.pipe.vae.config.latents_mean).view(1, self.pipe.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
        )
        latents_std = 1.0 / torch.tensor(self.pipe.vae.config.latents_std).view(
            1, self.pipe.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.pipe.vae.decode(latents, return_dict=False)[0]
        videos = self.pipe.video_processor.postprocess_video(video, output_type="pil")
        self._set_active_module(None)
        stage_durations = dict(state.extra["stage_durations"])
        stage_durations["decode"] = time.perf_counter() - t0
        image = state.extra["image"]
        return OmniOutput(
            request_id=state.req_id,
            frames=videos[0],
            width=image.width,
            height=image.height,
            fps=state.sampling.fps,
            stage_durations=stage_durations,
        )
