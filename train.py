import logging
import math
import os
import transformers
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

from train_config import train_args
from utils import save_progress, spherical_dist_loss, TextualInversionDataset


logger = get_logger(__name__)

def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    if args.report_to == "wandb":
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    # Load scheduler and models
    text_encoder = CLIPTextModelWithProjection.from_pretrained(args.clip_model_path).to(accelerator.device)
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2").to(accelerator.device)

    for tokenizr, text_enc in zip([tokenizer, tokenizer2], [text_encoder, text_encoder2]):
        with torch.no_grad():
            # Add the placeholder token in tokenizer
            if args.num_vectors == 1:
                num_added_tokens = tokenizr.add_tokens(args.placeholder_token)
                subtokens = None
            else:
                num_added_tokens = 0
                subtokens = [f'xyzzyx{i}' for i in range(args.num_vectors)]
                for tok in subtokens:
                    num_added_tokens = tokenizr.add_tokens(tok)

            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
                    " `placeholder_token` that is not already in the tokenizer."
                )

            if args.initializer_token is not None:
                # Convert the initializer_token, placeholder_token to ids
                initializer_token_id = tokenizr(args.initializer_token, add_special_tokens=False).input_ids
                # Check if initializer_token is a single token or a sequence of tokens
                if len(initializer_token_id) > 1:
                    raise ValueError("The initializer token must be a single token.")

                initializer_token_id = initializer_token_id

            if args.num_vectors > 1:
                initializer_token_id = [initializer_token_id] * args.num_vectors

                placeholder_token_id = tokenizr.convert_tokens_to_ids(subtokens)
            else:
                placeholder_token_id = tokenizr.convert_tokens_to_ids(args.placeholder_token)

            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            text_enc.resize_token_embeddings(len(tokenizr))

            # Initialise the newly added placeholder token with the embeddings of the initializer token
            token_embeds = text_enc.get_input_embeddings().weight.data
            if args.initializer_token is not None:
                if args.num_vectors > 1:
                    for i in range(args.num_vectors):
                        token_embeds[placeholder_token_id[i]] = token_embeds[initializer_token_id[i]]
                else:
                    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
            else:
                if args.num_vectors > 1:
                    for i in range(args.num_vectors):
                        token_embeds[placeholder_token_id[i]] = torch.randn_like(token_embeds[0]) * 0.01
                else:
                    token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[0]) * 0.01

            # Freeze all parameters except for the token embeddings in text encoder
            text_enc.text_model.encoder.requires_grad_(False)
            text_enc.text_model.final_layer_norm.requires_grad_(False)
            text_enc.text_model.embeddings.position_embedding.requires_grad_(False)
            text_enc.get_input_embeddings().weight.requires_grad_(True)

            if args.gradient_checkpointing:
                # Keep unet in train mode if we are using gradient checkpointing to save memory.
                # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
                text_enc.gradient_checkpointing_enable()


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        [text_encoder.get_input_embeddings().parameters(), text_encoder2.get_input_embeddings().parameters()],
        lr=args.clip_train_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        vision_model1_path=args.clip_model_path,
        #vision_model2=vision_model,
        initializer_token=args.initializer_token,
        pad_tokens=args.pad_tokens,
        subtokens=subtokens,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.clip_train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Scheduler and math around the number of training steps.
    clip_train_epochs = math.ceil(args.clip_max_train_steps / len(train_dataloader))

    lr_scheduler = get_scheduler(
        args.clip_lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.clip_max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    logger.info("***** Running CLIP training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.clip_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.clip_max_train_steps}")
    clip_global_step = 0

    # keep original embeddings as reference
    orig_embeds_params1 = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    orig_embeds_params2 = accelerator.unwrap_model(text_encoder2).get_input_embeddings().weight.data.clone()

    # assume 49407 max
    index_no_updates = torch.arange(len(tokenizer)) <= 49407

    if args.clip_phase_gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        text_encoder2.gradient_checkpointing_enable()

    pbar = tqdm(range(args.clip_max_train_steps))
    pbar.set_description("Steps")
    for epoch in range(clip_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step

            # Get the text embedding for conditioning
            cond_tok_ids = batch["input_ids"].to(accelerator.device)
            im_embed = batch["clip_embeds1"].to(accelerator.device).to(dtype=torch.float32).detach()
            im_embed2 = batch["clip_embeds2"].to(accelerator.device).to(dtype=torch.float32).detach()
            text_embeds = text_encoder(cond_tok_ids).last_hidden_state.to(dtype=torch.float32)
            text_embeds2 = text_encoder2(cond_tok_ids).last_hidden_state.to(dtype=torch.float32)

            # create a mask for indexing the EOF token, as Okaris noted, adding new tokens to vocab throws this process off
            # zero out all non-eof tokens
            mask = torch.where(cond_tok_ids==49407,cond_tok_ids,0)
            # this is how we'll index the pooler output
            text_embeds = text_embeds[torch.arange(text_embeds.shape[0]), mask.argmax(dim=-1)]
            text_embeds = text_encoder.text_projection(text_embeds)
            text_embeds2 = text_embeds2[torch.arange(text_embeds2.shape[0]), mask.argmax(dim=-1)]
            text_embeds2 = text_encoder2.text_projection(text_embeds2)

            if args.spherical_clip_loss == True:
                loss = spherical_dist_loss(text_embeds, im_embed)
                loss = loss + spherical_dist_loss(text_embeds2, im_embed2)
            else:
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                im_embed = im_embed / im_embed.norm(p=2, dim=-1, keepdim=True)
                sim = -torch.matmul(text_embeds, im_embed.t())
                #only similarity with correct pairings needed
                if args.zero_out_mismatches:
                    mask = torch.diag(torch.ones(sim.shape[0])).float().to(sim.device)
                    sim = sim * mask
                loss = sim.mean()

                text_embeds2 = text_embeds2 / text_embeds2.norm(p=2, dim=-1, keepdim=True)
                im_embed2 = im_embed2 / im_embed2.norm(p=2, dim=-1, keepdim=True)
                sim2 = -torch.matmul(text_embeds2, im_embed2.t())
                #only similarity with correct pairings needed
                if args.zero_out_mismatches:
                    mask = torch.diag(torch.ones(sim2.shape[0])).float().to(sim2.device)
                    sim2 = sim2 * mask
                loss = loss + sim2.mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if args.clip_max_grad_norm is not None:
                    accelerator.clip_grad_norm_(text_encoder.get_input_embeddings().parameters(), args.clip_max_grad_norm)
                    accelerator.clip_grad_norm_(text_encoder2.get_input_embeddings().parameters(), args.clip_max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            logs = {"loss": loss.mean().detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            pbar.set_postfix(**logs)
            pbar.set_description(f"Epoch {epoch}")
            pbar.update(1)

            # Let's make sure we don't update any embedding weights besides the newly added token
            with torch.no_grad():
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[index_no_updates] = orig_embeds_params1[index_no_updates]
                accelerator.unwrap_model(text_encoder2).get_input_embeddings().weight[index_no_updates] = orig_embeds_params2[index_no_updates]

            clip_global_step += 1
            if clip_global_step >= args.clip_max_train_steps:
                break


    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        vision_model1_path=None,
        vision_model2_path=None,
        initializer_token=args.initializer_token,
        pad_tokens=args.pad_tokens,
        vae_path=args.pretrained_model_name_or_path if args.cache_latents else None,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Scheduler and math around the number of training steps.
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)


    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # remake the optimizer
    optimizer = torch.optim.AdamW(
        [text_encoder.get_input_embeddings().parameters(), text_encoder2.get_input_embeddings().parameters()],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, text_encoder2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, text_encoder2, optimizer, train_dataloader, lr_scheduler
    )

    if args.cache_latents == False:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        # Freeze vae and unet
        vae.requires_grad_(False)
        vae.to(accelerator.device, dtype=weight_dtype)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()


    # Move vae and unet to device and cast to weight_dtype
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")



    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(
            add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        add_time_ids = torch.tensor([add_time_ids])
        return add_time_ids

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # second wave
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(text_encoder):
                with torch.no_grad():
                    if args.cache_latents == False:
                        # Convert images to latent space
                        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                        latents = latents * 0.18125
                    else:
                        latents = batch["latents"].to(dtype=weight_dtype).detach()

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states1 = text_encoder(batch["input_ids"]).hidden_states[-1].to(dtype=weight_dtype)
                output = text_encoder2(batch["input_ids"])
                encoder_hidden_states2 = output.hidden_states[-1].to(dtype=weight_dtype)
                pooled = output.pooler_output.to(dtype=weight_dtype)
                encoder_hidden_states = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2)

                original_size = (noisy_latents.shape[-2]*8, noisy_latents.shape[-1]*8)
                target_size = (noisy_latents.shape[-2]*8, noisy_latents.shape[-1]*8)
                crops_coords_top_left = (0, 0)
                add_time_ids = _get_add_time_ids(unet, original_size, crops_coords_top_left, target_size).to("cuda").to(weight_dtype).repeat(noisy_latents.shape[0], 1)

                added_cond_kwargs = {"text_embeds": pooled, "time_ids": add_time_ids}

                # Predict the noise residual
                with torch.no_grad():
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                                      added_cond_kwargs=added_cond_kwargs,).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(
                            text_encoder.get_input_embeddings().parameters(),
                            args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params1[index_no_updates]
                    accelerator.unwrap_model(text_encoder2).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params2[index_no_updates]


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path, subtokens, logger)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.mean().detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
            logger.info(
                f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                f" {args.validation_prompt}."
            )
            # create pipeline (note: unet and vae are loaded again in float32)
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                torch_dtype=weight_dtype,
            )
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            generator = (
                None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
            )
            images = []
            for _ in range(args.num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                images.append(image)

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "validation": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

            del pipeline
            torch.cuda.empty_cache()

    # save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "learned_embeds_vitl.bin")
        save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path, subtokens, logger)
        save_path = os.path.join(args.output_dir, "learned_embeds_vitg.bin")
        save_progress(text_encoder2, placeholder_token_id, accelerator, args, save_path, subtokens, logger)

    accelerator.end_training()

if __name__ == "__main__":
    main(train_args)