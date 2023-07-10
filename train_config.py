from types import SimpleNamespace
train_args = dict({
"save_steps": 500,
"clip_model_path": "openai/clip-vit-large-patch14",
"pretrained_model_name_or_path": "Lykon/DreamShaper", #finetuned models are less likely to have a good sense of your concept, just keep in mind
"revision": None,
"tokenizer_name": None,
"train_data_dir": "./traindata",
"placeholder_token": "sksts",
"initializer_token": "person",
"learnable_property": "object",
"repeats": 100,
"output_dir": "text-inversion-model",
"seed": None,
"resolution": 512,
"center_crop": True,
"clip_train_batch_size": 32,
"clip_max_train_steps": 200,
"clip_train_lr": 9.9e-02, # lower this if its overfitting
"clip_phase_gradient_checkpointing": False, # turn on for more mem savings
"clip_lr_scheduler": "linear",
"spherical_clip_loss": False,
"clip_max_grad_norm": 1.5,
"max_train_steps": 50,
"gradient_accumulation_steps": 1,
"gradient_checkpointing": True,
"train_batch_size": 4,
"learning_rate": 1.5e-04,
"scale_lr": False,
"lr_scheduler": "constant",
"lr_warmup_steps": 0,
"dataloader_num_workers": 0, # set to 0 if this causes issues
"adam_beta1": 0.9,
"adam_beta2": 0.999,
"adam_weight_decay": 1e-2,
"adam_epsilon": 1e-08,
"logging_dir": "logs",
"mixed_precision": "bf16",
"allow_tf32": True,
"report_to": "tensorboard",
"validation_prompt": None,
"num_validation_images": 4,
"validation_epochs": 50,
"local_rank": -1,
"checkpointing_steps": 5000,
"resume_from_checkpoint": None,
"enable_xformers_memory_efficient_attention": True,
"vision_model_pretrained": "openai/clip-vit-large-patch14",
"zero_out_mismatches":True,
"what_to_train": "layer",
"max_grad_norm": 1,
"num_vectors": 1,
"pad_tokens": True,
"cache_latents": True,
})

train_args = SimpleNamespace(**train_args)






