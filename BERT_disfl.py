import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    RobertaTokenizerFast, 
    RobertaForTokenClassification,
    RobertaModel,
    PreTrainedModel,
    set_seed,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForTokenClassification
)
import os
import random
import sys
from switchboard_disfl import get_switchboard_disfluency_dataset
from text_utils import get_conversation_ids_from_file
from audio_utils import SWB_ROOT, PROJECT_ROOT
from datasets import DatasetDict, load_metric, Dataset
import transformers
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, logging
from transformers.utils.versions import require_version
import torch
import wandb

wandb.init(project="bert-disfl")

def main():
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    logger = logging.get_logger()

    training_args = TrainingArguments(
        output_dir="trained_models/bert_disfl",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=15,
        do_train=True
    )

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
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)


    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()

    set_seed(training_args.seed)

    raw_datasets = DatasetDict()

    train_conversation_ids_path = os.path.join(SWB_ROOT, 'splits', 'ws97-train-convs.list')
    train_conversation_ids = get_conversation_ids_from_file(train_conversation_ids_path)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

    switchboard_df = get_switchboard_disfluency_dataset(train_conversation_ids)

    dataset = Dataset.from_pandas(switchboard_df)
    raw_datasets = dataset.train_test_split(test_size=0.1, seed=training_args.seed)

    rand_indx = random.randint(0, len(raw_datasets["train"]))
    print(raw_datasets["train"][rand_indx])

    label_all_tokens = True

    def tokenize_and_align_dataset(batch):
        tokenized_inputs = tokenizer.batch_encode_plus(
            batch['text'], 
            truncation=True, 
            # padding=True, 
            # return_tensors='pt', 
            is_split_into_words=True
        )
        # logger.info(f"tokenized_inputs {tokenized_inputs}\n\n")

        labels = []
        for i, label in enumerate(batch['tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            # logger.info(f"label: {label}\n\n")
            # logger.info(f"word_ids: {word_ids}")
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return {
            "input_ids": tokenized_inputs['input_ids'],
            "attention_mask": tokenized_inputs['attention_mask'],
            "labels": tokenized_inputs["labels"]
        }
    
    tokenized_ds = raw_datasets.map(
        tokenize_and_align_dataset, 
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        )
    logger.info(tokenized_ds.shape)
    logger.info(tokenized_ds["train"][rand_indx])
    logger.info(tokenized_ds["train"].features)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=2)


    metric = load_metric("seqeval")

    label_list = [0,1]

    def compute_metrics(eval_preds):
        logger.info(f"eval_preds: {eval_preds}\n\n")
        logits, labels = eval_preds
        logger.info(f"logits: {logits}\n")
        logger.info(f"labels: {labels}\n")
        preds = np.argmax(logits, axis=2)

        true_labels = [[labels[l] for l in label if l != -100] for label in labels]
        
        true_preds = [
            [label_list[p] for (p,l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]


        results = metric.compute(preds=true_preds, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    model_name_or_path = "roberta-base"

    # use last checkpoint if exist
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_name_or_path):
        checkpoint = model_name_or_path
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

if __name__ == "__main__":
    main()