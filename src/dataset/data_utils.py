import logging
from pathlib import Path
from typing import Generator, Optional

from datasets import Dataset, concatenate_datasets, disable_caching, load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

disable_caching()

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent
logger = logging.getLogger(__name__)


def build_part(dataset_params: dict[str, str]) -> Generator[dict[str, str], None, None]:
    """
    Iterator to part of large dataset

    Args:
        dataset_params (dict[str, str]): Dataset part configuration

    Yields:
        Generator[dict[str, str], None, None]: Yield entry of dataset
    """
    max_samples = int(dataset_params.max_samples)
    dataset = load_dataset(**dataset_params.params, streaming=True)
    for idx, example in tqdm(enumerate(dataset), total=max_samples):
        if idx >= max_samples:
            return
        yield example


def build_dataset(dataset_cfg: DictConfig, base_path: Path | str = ROOT_PATH) -> None:
    """
    Download and build concatenated dataset for pretrain stage from config

    Args:
        dataset_cfg (DictConfig): Сonfiguration of dataset parts
        base_path (Path | str, optional): Save path of new dataset. Defaults to ROOT_PATH.
    """
    base_path = Path(base_path).absolute().resolve()
    save_dir = base_path / dataset_cfg.build_params.save_dir
    to_concatenation = []

    logger.info("Start building dataset")
    # load datasets and source code
    for dataset_name, dataset_params in tqdm(dataset_cfg.datasets.items()):
        logger.info(f"Process dataset {dataset_name}")
        if dataset_params.get("dataset_part", None) is not None:
            file_pattern = (
                dataset_params.dataset_part.base_url
                + dataset_params.dataset_part.file_pattern
            )
            start_idx = dataset_params.dataset_part.start_idx
            end_idx = dataset_params.dataset_part.end_idx
            data_files = {
                "train": [
                    file_pattern.format(i=idx) for idx in range(start_idx, end_idx)
                ]
            }
            dataset = load_dataset(
                **dataset_params.params,
                data_files=data_files,
                num_proc=dataset_cfg.build_params.num_proc,
            )
        elif dataset_params.get("max_samples", None) is not None:
            dataset = Dataset.from_generator(
                build_part,
                gen_kwargs={"dataset_params": dataset_params},
            )
        else:
            dataset = load_dataset(
                **dataset_params.params, num_proc=dataset_cfg.build_params.num_proc
            )

        if dataset_params.get("rename_column", None) is not None:
            dataset = dataset.rename_column(**dataset_params.rename_column)
        if dataset_params.get("select_columns", None) is not None:
            dataset = dataset.select_columns(dataset_params.select_columns)
        to_concatenation.append(dataset)

    dataset = concatenate_datasets(to_concatenation)
    dataset = dataset.train_test_split(
        test_size=dataset_cfg.build_params.test_size,
        seed=dataset_cfg.build_params.get("seed", None),
    )
    logger.info(f"Saving prepared dataset to {save_dir}")
    dataset.save_to_disk(
        save_dir,
        max_shard_size="2GB",
        num_proc=dataset_cfg.build_params.num_proc,
    )


def apply_conversational_format(
    example: dict[str, str],
    process_data: Optional[dict[str, str]],
    chat_col: Optional[str],
) -> dict[str, list[dict[str, str]]]:
    """
    Transform dataset for instruction tuning from from an arbitrary format
    to conversational format

    Args:
        example (dict[str, str]): Entry of dataset for conversational transform
        process_data (Optional[dict[str, str]]): Map to role columns in entry
        chat_col (Optional[str]): Map to column already in entry if exists

    Returns:
        dict[str, list[dict[str, str]]]: Conversational formated entry of dataset
    """
    if process_data is not None:
        chat = []
        if process_data.get("system_col") is not None:
            chat.append({"role": "system", "content": example[process_data.system_col]})
        chat.extend(
            [
                {"role": "user", "content": example[process_data.user_col]},
                {"role": "assistant", "content": example[process_data.assistant_col]},
            ]
        )
    else:
        chat = example[chat_col]
    return {"messages": chat}


def build_instruct_dataset(
    dataset_cfg: DictConfig, base_path: Path | str = ROOT_PATH
) -> None:
    """
    Download and build concatenated dataset for instruction sft stage from config

    Args:
        dataset_cfg (DictConfig): Сonfiguration of data set parts
        base_path (Path | str, optional): Save path of new dataset. Defaults to ROOT_PATH.
    """
    base_path = Path(base_path).absolute().resolve()
    save_dir = base_path / dataset_cfg.build_params.save_dir
    to_concatenation = []

    logger.info("Start building instruction dataset")
    # load datasets and source code
    for dataset_name, dataset_params in tqdm(dataset_cfg.datasets.items()):
        logger.info(f"Process dataset {dataset_name}")
        dataset = load_dataset(
            **dataset_params.params, num_proc=dataset_cfg.build_params.num_proc
        )
        dataset = dataset.map(
            apply_conversational_format,
            fn_kwargs={
                "process_data": dataset_params.get("process_data", None),
                "chat_col": dataset_params.get("chat_col", "messages"),
            },
        ).select_columns("messages")

        to_concatenation.append(dataset)
    dataset = concatenate_datasets(to_concatenation)

    dataset = dataset.train_test_split(
        dataset_cfg.build_params.test_size,
        seed=dataset_cfg.build_params.get("seed", None),
    )
    logger.info(f"Saving prepared dataset to {save_dir}")
    dataset.save_to_disk(
        save_dir,
        max_shard_size="2GB",
        num_proc=dataset_cfg.build_params.num_proc,
    )
