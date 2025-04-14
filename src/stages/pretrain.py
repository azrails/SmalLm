import logging
import os
import sys
import warnings
from pathlib import Path

import datasets
import hydra
import torch.nn as nn
import transformers
from datasets import load_from_disk
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich import print
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent
os.environ["HYDRA_FULL_ERROR"] = "1"
sys.path.insert(1, ".")


from src.models import SmalLmConfig, SmalLmForCausalLM, SmalLmModel
from src.models.model import Router


class AdjustLRCallback(TrainerCallback):
    def __init__(self, new_lr):
        self.new_lr = new_lr

    def on_train_begin(self, args, state, control, **kwargs):
        optimizer = kwargs["optimizer"]
        scheduler = kwargs["lr_scheduler"]

        # Обновляем learning rate у оптимизатора
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.new_lr

        # Обновляем scheduler
        scheduler.base_lrs = [self.new_lr]

        logger.info(f"Updated learning rate to {self.new_lr}")


class BiasUpdateCallback(TrainerCallback):
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs,
    ):
        for module in model.modules():
            if isinstance(module, Router):
                module.update_bias()
        return super().on_step_end(args, state, control, **kwargs)


class BeforeClippingGradientLoggingCallback(TrainerCallback):
    def __init__(self, model, log_interval=10):
        self.log_interval = log_interval  # Логировать каждые N шагов
        self.model = model

    def on_substep_end(self, args, state, control, model, **kwargs):
        if state.global_step % self.log_interval == 0:
            logs = {}
            total_norm = 0.0

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.detach().norm(2).item()
                    logs[f"grad_norm/{name}"] = grad_norm
                    total_norm += grad_norm**2

            logs["grad_norm/total"] = total_norm**0.5

            i = 1
            for module in model.modules():
                if isinstance(module, Router):
                    logs[f"bias MoE layer: {i}"] = module.bias
                    i += 1
            print(logs)


@hydra.main(config_path="../configs/pretrain", config_name="SmalLm_260M")
def main(cfg: DictConfig):
    set_seed(cfg.get("seed", None))
    output_dir = ROOT_PATH / Path(cfg.training_arguments.output_dir)
    training_arguments: TrainingArguments = instantiate(
        cfg.training_arguments, output_dir=output_dir
    )

    #################
    # Set Logging
    #################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_arguments.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(
        f"Process rank: {training_arguments.local_rank}, device: {training_arguments.device}, n_gpu: {training_arguments.n_gpu}"
        + f" distributed training: {bool(training_arguments.local_rank != -1)}"
    )

    #################
    # Configure model
    #################
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer, use_fast=True, trust_remote_code=True
    )
    model_config = instantiate(cfg.model_config)
    model = SmalLmForCausalLM(config=model_config)

    #################
    # Load data
    #################
    def apply_tokenizer(examples, max_length):
        text = examples["text"]
        out = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {"input_ids": out.input_ids, "attention_mask": out.attention_mask}

    dataset = load_from_disk(ROOT_PATH / cfg.dataset_dir)
    train_dataset = (
        dataset["train"]
        .map(
            apply_tokenizer,
            batched=True,
            fn_kwargs={"max_length": cfg.model_config.max_seq_len},
            batch_size=10000,
            num_proc=32,
        )
        .select_columns(["input_ids", "attention_mask"])
    )
    test_dataset = (
        dataset["test"]
        .select(range(5000))
        .map(
            apply_tokenizer,
            batched=True,
            fn_kwargs={"max_length": cfg.model_config.max_seq_len},
            batch_size=10000,
            num_proc=32,
        )
        .select_columns(["input_ids", "attention_mask"])
    )

    #################
    # Configure training
    #################
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, return_tensors="pt"
    )

    callbacks = [
        BiasUpdateCallback(),
    ]  # , BeforeClippingGradientLoggingCallback(model)],
    checkpoint = None
    if training_arguments.resume_from_checkpoint is not None and not isinstance(
        training_arguments.resume_from_checkpoint, bool
    ):
        checkpoint = training_arguments.resume_from_checkpoint
    elif isinstance(training_arguments.resume_from_checkpoint, bool):
        checkpoint = training_arguments.resume_from_checkpoint

    if checkpoint is not None:
        callbacks.append(AdjustLRCallback(training_arguments.learning_rate))

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    logger.info("Start training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()

    #################
    # Saving model
    #################
    trainer.save_model(training_arguments.output_dir)
    logger.info(f"Model saved to {training_arguments.output_dir}")
    kwargs = {
        "tags": [f"{cfg.model_name}"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.save_pretrained(training_arguments.output_dir)

    #################
    # Register for hub
    #################
    AutoConfig.register("smallm", SmalLmConfig)
    AutoModel.register(SmalLmConfig, SmalLmModel)
    AutoModelForCausalLM.register(SmalLmConfig, SmalLmForCausalLM)
    SmalLmConfig.register_for_auto_class()
    SmalLmModel.register_for_auto_class("AutoModel")
    SmalLmForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = AutoModelForCausalLM.from_pretrained(f"{training_arguments.output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(f"{training_arguments.output_dir}")
    model.save_pretrained(f"{training_arguments.output_dir}")
    tokenizer.save_pretrained(f"{training_arguments.output_dir}")

    if training_arguments.push_to_hub:
        logger.info("Push to hub...")
        model.push_to_hub(f"{cfg.model_name}_{cfg.model_suffix}", **kwargs)
        tokenizer.push_to_hub(f"{cfg.model_name}_{cfg.model_suffix}", **kwargs)


if __name__ == "__main__":
    main()
