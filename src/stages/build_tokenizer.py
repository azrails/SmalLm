import os
from pathlib import Path

import hydra
import tokenizers
from datasets import load_from_disk
from omegaconf import DictConfig
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

os.environ["HYDRA_FULL_ERROR"] = "1"


def get_training_corpus(dataset, batch_size: int = 10000):
    dataset = dataset.select_columns("text")
    for batch in dataset.iter(batch_size):
        yield batch["text"]


@hydra.main(config_path="../configs", config_name="tokenizer", version_base=None)
def main(cfg: DictConfig):
    # load train dataset
    data_dir = Path(cfg.data_dir).absolute().resolve()
    dataset = load_from_disk(data_dir)["train"]

    # initialize base tokenizer
    tokenizer = Tokenizer(models.BPE(cache_capacity=cfg.cache_capacity))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(tokenizers.Regex(r"\s+"), content=" "),
            normalizers.Strip(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    special_tokens = [token_name for _, token_name in cfg.special_tokens.items()] + [
        f"<|reserved_token_{i + 1}|>" for i in range(cfg.num_reserved_tokens)
    ]
    tokenizer.decoder = decoders.ByteLevel()

    # train tokenizer
    trainer = trainers.BpeTrainer(
        **cfg.trainer_params,
        show_progress=True,
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(
        get_training_corpus(dataset, cfg.tainer_batch_size),
        trainer=trainer,
        length=len(dataset),
    )

    eos_id = tokenizer.token_to_id(cfg.special_tokens.eos_token)
    tokenizer.post_processor = processors.Sequence(
        [
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(
                single=f"$A:0 {cfg.special_tokens.eos_token}:0",
                special_tokens=[(f"{cfg.special_tokens.eos_token}", eos_id)],
            ),
        ]
    )

    # build fast tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, **cfg.special_tokens
    )

    # save & push
    save_dir = Path(cfg.save_dir).absolute().resolve()
    fast_tokenizer.save_pretrained(save_dir)
    if cfg.get("push_to_hub", None) is not None:
        repo_path = cfg.push_to_hub
        fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(save_dir)
        tokenizer = PreTrainedTokenizer.from_pretrained(save_dir)
        fast_tokenizer.push_to_hub(repo_path)


if __name__ == "__main__":
    main()
