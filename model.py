import json
import re
import functools
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from audio_utils import PROJECT_ROOT, SWB_ROOT, get_conversation_slice
from text_utils import get_conversation_ids_from_file
from switchboard_disfl import get_switchboard_disfluency_dataset
from datasets import DatasetDict, load_metric, Dataset, Audio
from typing import Dict, List, Optional, Union
import numpy as np
import librosa
import wandb

wandb.init(project="hubert-disfl")

import transformers
from transformers import (
    HfArgumentParser, 
    TrainingArguments, 
    Trainer,
    Wav2Vec2Processor, 
    HubertForCTC, 
    HubertConfig, 
    set_seed)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import torch

from switchboard_disfl import SwitchboardDisfluencyDataset

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.19.0.dev0")

logger = logging.getLogger(__name__)

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    attention_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    feat_proj_dropout: float = field(default=0.0, metadata={"help": "The dropout ratio for the projected features."})
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": "Probability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis."
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability of each feature vector along the feature axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature bins will be masked along the time axis."
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: Optional[str] = field(
        default="mean", metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # dataset_name: str = field(
    #     metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    # )
    dataset_config_name: str = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: str = field(
        default="train+validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to "
            "'train+validation'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="target_text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    chars_to_ignore: Optional[List[str]] = list_field(
        default=None,
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    eval_metrics: List[str] = list_field(
        default=["wer"],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": "Filter audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only do data preprocessing and skip training. "
            "This is especially useful when data preprocessing errors out in distributed training due to timeout. "
            "In this case, one should run the preprocessing in a non-distributed setup with `preprocessing_only=True` "
            "so that the cached datasets can consequently be loaded in distributed training"
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "If :obj:`True`, will use the token generated when running"
            ":obj:`transformers-cli login` as HTTP bearer authorization for remote files."
        },
    )
    unk_token: str = field(
        default="[UNK]",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: str = field(
        default="[PAD]",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
    )
    phoneme_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "The target language that should be used be"
            " passed to the tokenizer for tokenization. Note that"
            " this is only relevant if the model classifies the"
            " input audio to a sequence of phoneme sequences."
        },
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch["target_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names,
    )

    # take union of all unique characters in each dataset
    vocab_set = functools.reduce(
        lambda vocab_1, vocab_2: set(vocab_1["vocab"][0]) | set(vocab_2["vocab"][0]), vocabs.values()
    )

    vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()

    set_seed(training_args.seed)

    max_input_length = data_args.max_duration_in_seconds * 16_000
    min_input_length = data_args.min_duration_in_seconds * 16_000
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers

    raw_datasets = DatasetDict()

    if training_args.do_train:
        train_conversation_ids_path = os.path.join(SWB_ROOT, 'splits', 'ws97-train-convs.list')
        train_conversation_ids = get_conversation_ids_from_file(train_conversation_ids_path)[75:100]
        # train_conversation_ids = train_conversation_ids[:1]
        switchboard_df = get_switchboard_disfluency_dataset(
            train_conversation_ids, 
            16_000, 
            chars_to_ignore=data_args.chars_to_ignore, 
            min_length=data_args.min_duration_in_seconds, 
            max_length=data_args.max_duration_in_seconds,
            fluent=True
        )
        print(switchboard_df)
        raw_datasets["train"] = Dataset.from_pandas(switchboard_df)

    logger.info("Training/evaluation parameters %s", training_args)

    text_column_name = data_args.text_column_name
    # chars_to_ignore_regex = (
    #     f'[{"".join(data_args.chars_to_ignore)}]' if data_args.chars_to_ignore is not None else None
    # )

    # def remove_special_characters(batch):
    #     if chars_to_ignore_regex is not None:
    #         batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower() + " "
    #     else:
    #         batch["target_text"] = batch[text_column_name].lower() + " "
    #     return batch
    
    # with training_args.main_process_first(desc="dataset map special characters removal"):
    #     raw_datasets = raw_datasets.map(
    #         remove_special_characters,
    #         remove_columns=[text_column_name],
    #         desc="remove special characters from datasets",
    #     )

    # word_delimiter_token = data_args.word_delimiter_token
    # unk_token = data_args.unk_token
    # pad_token = data_args.pad_token

    # 3. Next, let's load the config as we might need it to create
    # the tokenizer
    # load config
    # config = HubertConfig()

    # 4. Next, if no tokenizer file is defined,
    # we create the vocabulary of the model by extracting all unique characters from
    # the training and evaluation datasets
    # We need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    # tokenizer_name_or_path = model_args.tokenizer_name_or_path
    # tokenizer_kwargs = {}
    # if tokenizer_name_or_path is None:
    #     # save vocab in training output dir
    #     tokenizer_name_or_path = training_args.output_dir

    #     vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")

    #     with training_args.main_process_first():
    #         if training_args.overwrite_output_dir and os.path.isfile(vocab_file):
    #             os.remove(vocab_file)

    #     with training_args.main_process_first(desc="dataset map vocabulary creation"):
    #         if not os.path.isfile(vocab_file):
    #             os.makedirs(tokenizer_name_or_path, exist_ok=True)
    #             vocab_dict = create_vocabulary_from_data(
    #                 raw_datasets,
    #                 word_delimiter_token=word_delimiter_token,
    #                 unk_token=unk_token,
    #                 pad_token=pad_token,
    #             )

    #             # save vocab dict to be loaded into tokenizer
    #             with open(vocab_file, "w") as file:
    #                 json.dump(vocab_dict, file)

    #     # if tokenizer has just been created
    #     # it is defined by `tokenizer_class` if present in config else by `model_type`
    #     tokenizer_kwargs = {
    #         "config": config if config.tokenizer_class is not None else None,
    #         "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
    #         "unk_token": unk_token,
    #         "pad_token": pad_token,
    #         "word_delimiter_token": word_delimiter_token,
    #     }

    processor = Wav2Vec2Processor.from_pretrained('facebook/hubert-large-ls960-ft')
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()
    
    # raw_datasets["train"] = raw_datasets["train"].cast_column("audio", Audio(sampling_rate=16_000))

    def prepare_dataset(batch):
        assert batch["duration"] > 0
        sample = get_conversation_slice(batch["conversation_id"], batch["start_time"], batch["end_time"])

        batch["input_values"] = processor(sample, sampling_rate=16_000).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch[text_column_name]).input_ids
        return batch
        
    for i in range(len(raw_datasets["train"])):
        print(raw_datasets["train"][i])
    print()
    lengths = [raw_datasets["train"][i]["duration"] for i in range(len(raw_datasets["train"]))]
    print(max(lengths), min(lengths))
    
    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=num_workers,
            desc="preprocess datasets",
            batched=True
        )

        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        # filter data that is shorter than min_input_length
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_length"],
        )

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return
    
    lengths = [vectorized_datasets["train"][i]["input_length"]/16_000 for i in range(len(vectorized_datasets["train"]))]
    print(max(lengths), min(lengths))

    eval_metrics = {metric: load_metric(metric) for metric in data_args.eval_metrics}
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics

    # Now save everything to be able to create a single processor later
    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        processor.save_pretrained(training_args.output_dir)

    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # print(vectorized_datasets["train"][0])
    # print(vectorized_datasets["train"][0].keys())
    # return

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        # eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )

    if training_args.do_train:

        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # dataset=dataset.sort("id")
    # sampling_rate = dataset.features["audio"].sampling_rate

    # # audio file is decoded on the fly
    # inputs = processor(dataset[1]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    # with torch.no_grad():
    #     logits = model(**inputs).logits

    # predicted_ids = torch.argmax(logits, dim=-1)

    # # transcribe speech
    # transcription = processor.batch_decode(predicted_ids)
    # print(transcription[0])

if __name__ == "__main__":
    main()
