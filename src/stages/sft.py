import logging
import os
import sys
import warnings
from pathlib import Path

import datasets
import hydra
import transformers
from datasets import load_from_disk
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent
os.environ["HYDRA_FULL_ERROR"] = "1"
sys.path.insert(1, ".")

from src.models import SmalLmConfig, SmalLmForCausalLM, SmalLmModel


@hydra.main(config_path="../configs/sft", config_name="SmalLm-70", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.get("seed", None))
    output_dir = ROOT_PATH / Path(cfg.training_arguments.output_dir)
    training_arguments: SFTConfig = instantiate(
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
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True, **cfg.model_config
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_tokem = tokenizer.eos_token

    #################
    # Load data
    #################
    dataset = load_from_disk(ROOT_PATH / cfg.dataset_dir)

    #################
    # Configure training
    #################
    checkpoint = None
    if training_arguments.resume_from_checkpoint is not None and not isinstance(
        training_arguments.resume_from_checkpoint, bool
    ):
        checkpoint = training_arguments.resume_from_checkpoint
    elif isinstance(training_arguments.resume_from_checkpoint, bool):
        checkpoint = training_arguments.resume_from_checkpoint

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
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
