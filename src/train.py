# train.py
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import Dataset
from sklearn.metrics import roc_auc_score

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model

import config as CFG


class T5PromptLabelTrainer:
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        text_col: str,
        all_labels: list[str],
        model_name: str = "t5-base",
        selected_labels: list[str] | None = None,
        max_source_len: int = 256,
        max_target_len: int = 4,
        num_virtual_tokens: int = 10,
        prompt_init_text: str = "Answer the question with yes or no.",
        fp16: bool = True,
        seed: int = 42,
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.text_col = text_col
        self.all_labels = list(all_labels)
        self.selected_labels = selected_labels or list(all_labels)

        bad = [x for x in self.selected_labels if x not in self.all_labels]
        if bad:
            raise ValueError(f"selected_labels contains unknown labels: {bad}")

        self.model_name = model_name
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.num_virtual_tokens = num_virtual_tokens
        self.prompt_init_text = prompt_init_text
        self.fp16 = fp16
        self.seed = seed

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_id = self.tokenizer.encode("no", add_special_tokens=False)[0]

        self.model = None
        self.data_collator = None
        self.trainer = None

        self.train_tok = None
        self.val_tok = None
        self.test_tok = None

    def load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        need_cols = [self.text_col] + self.all_labels
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        return df

    def expand_df(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            text = str(r[self.text_col])
            for lab in self.selected_labels:
                y = int(r[lab])
                inp = f"{text}\n\nIs the text above {lab}?"
                tgt = "yes" if y == 1 else "no"
                rows.append({"text": inp, "target": tgt, "y": y, "label_name": lab})
        return pd.DataFrame(rows)

    def preprocess(self, batch):
        model_inputs = self.tokenizer(
            batch["text"],
            max_length=self.max_source_len,
            truncation=True,
            padding=False,
        )
        labels_tok = self.tokenizer(
            text_target=batch["target"],
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
        )["input_ids"]
        model_inputs["labels"] = labels_tok
        model_inputs["y"] = batch["y"]
        model_inputs["label_name"] = batch["label_name"]
        return model_inputs

    def build_datasets(self):
        train_df = self.load_csv(self.train_path)
        val_df = self.load_csv(self.val_path)
        test_df = self.load_csv(self.test_path)

        train_exp = self.expand_df(train_df)
        val_exp = self.expand_df(val_df)
        test_exp = self.expand_df(test_df)

        train_ds = Dataset.from_pandas(train_exp.reset_index(drop=True))
        val_ds = Dataset.from_pandas(val_exp.reset_index(drop=True))
        test_ds = Dataset.from_pandas(test_exp.reset_index(drop=True))

        self.train_tok = train_ds.map(self.preprocess, batched=True, remove_columns=train_ds.column_names)
        self.val_tok   = val_ds.map(self.preprocess,   batched=True, remove_columns=val_ds.column_names)
        self.test_tok  = test_ds.map(self.preprocess,  batched=True, remove_columns=test_ds.column_names)

    def build_model(self):
        base_model = T5ForConditionalGeneration.from_pretrained(self.model_name)

        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text=self.prompt_init_text,
            num_virtual_tokens=self.num_virtual_tokens,
            tokenizer_name_or_path=self.model_name,
            tokenizer_kwargs={"use_fast": True},
        )
        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()

        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

    @staticmethod
    def _softmax2(a, b):
        m = np.maximum(a, b)
        ea = np.exp(a - m)
        eb = np.exp(b - m)
        return ea / (ea + eb)

    def compute_metrics(self, eval_pred):
        preds, _ = eval_pred
        first_step_logits = preds[:, 0, :]
        p_yes = self._softmax2(first_step_logits[:, self.yes_id], first_step_logits[:, self.no_id])

        ds = self.trainer._last_eval_dataset
        y_true = np.array(ds["y"])
        label_names = np.array(ds["label_name"])

        out = {}
        aucs = []
        for lab in self.selected_labels:
            mask = (label_names == lab)
            if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
                continue
            auc = roc_auc_score(y_true[mask], p_yes[mask])
            out[f"auc_{lab}"] = float(auc)
            aucs.append(auc)

        out["auc_macro"] = float(np.mean(aucs)) if aucs else 0.0
        return out

    def eval_auc_stream(self, dataset, split_name="val", batch_size=8):
        self.model.eval()

        def collate_for_model(features):
            keep = ["input_ids", "attention_mask", "labels"]
            features2 = [{k: f[k] for k in keep if k in f} for f in features]
            return self.data_collator(features2)

        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_for_model)

        p_yes_list = []
        with torch.no_grad():
            for batch in dl:
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)

                decoder_input_ids = torch.full(
                    (input_ids.size(0), 1),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=self.model.device,
                )

                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                logits0 = out.logits[:, 0, :]

                two = torch.stack([logits0[:, self.no_id], logits0[:, self.yes_id]], dim=-1)
                p_yes = torch.softmax(two, dim=-1)[:, 1]
                p_yes_list.append(p_yes.detach().cpu().numpy())

        p_yes = np.concatenate(p_yes_list, axis=0)
        y_true = np.array(dataset["y"])
        label_names = np.array(dataset["label_name"])

        out = {}
        aucs = []
        for lab in self.selected_labels:
            mask = (label_names == lab)
            if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
                continue
            auc = roc_auc_score(y_true[mask], p_yes[mask])
            out[f"auc_{lab}"] = float(auc)
            aucs.append(auc)

        out["auc_macro"] = float(np.mean(aucs)) if aucs else 0.0
        print(f"\n[{split_name.upper()}] auc_macro={out['auc_macro']:.4f}\n")
        return out

    class _AucTrainer(Seq2SeqTrainer):
        def evaluate(self, eval_dataset=None, **kwargs):
            self._last_eval_dataset = eval_dataset
            return super().evaluate(eval_dataset=eval_dataset, **kwargs)

    class _EvalEachEpochCallback(TrainerCallback):
        def __init__(self, outer, batch_size=8):
            self.outer = outer
            self.batch_size = batch_size

        def on_epoch_end(self, args, state, control, **kwargs):
            self.outer.eval_auc_stream(self.outer.val_tok, "val", batch_size=self.batch_size)
            return control

    def build_trainer(self, training_args: Seq2SeqTrainingArguments, eval_each_epoch=True, eval_bs=8):
        self.trainer = self._AucTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_tok,
            eval_dataset=self.val_tok,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        if eval_each_epoch:
            self.trainer.add_callback(self._EvalEachEpochCallback(self, batch_size=eval_bs))

    def run(self, training_args, eval_each_epoch=True, eval_stream_bs=8, save_dir="./adapter"):
        self.build_datasets()
        self.build_model()
        self.build_trainer(training_args, eval_each_epoch=eval_each_epoch, eval_bs=eval_stream_bs)

        print("=== TEST BEFORE TRAIN ===")
        self.eval_auc_stream(self.test_tok, "test_before", batch_size=eval_stream_bs)

        self.trainer.train()

        print("=== TEST AFTER TRAIN ===")
        self.eval_auc_stream(self.test_tok, "test_after", batch_size=eval_stream_bs)

        self.trainer.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Saved to {save_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", type=str, default=None, help="Train only this label (e.g. hostile)")
    p.add_argument("--labels", nargs="+", default=None, help="Train multiple labels: --labels hostile sarcastic")

    # override configs if needed
    p.add_argument("--num_virtual_tokens", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train_bs", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no_eval_each_epoch", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CFG.TrainConfig()

    # apply CLI overrides
    if args.num_virtual_tokens is not None:
        cfg.num_virtual_tokens = args.num_virtual_tokens
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.train_bs is not None:
        cfg.train_bs = args.train_bs
    if args.fp16:
        cfg.fp16 = True
    if args.no_eval_each_epoch:
        cfg.eval_each_epoch = False

    # resolve selected labels
    if args.labels is not None:
        selected = args.labels
    elif args.label is not None:
        selected = [args.label]
    else:
        selected = CFG.LABELS

    set_seed(cfg.seed)

    runner = T5PromptLabelTrainer(
        train_path=CFG.TRAIN_PATH,
        val_path=CFG.VAL_PATH,
        test_path=CFG.TEST_PATH,
        text_col=CFG.TEXT_COL,
        all_labels=CFG.LABELS,
        selected_labels=selected,
        model_name=cfg.model_name,
        num_virtual_tokens=cfg.num_virtual_tokens,
        prompt_init_text=cfg.prompt_init_text,
        max_source_len=cfg.max_source_len,
        max_target_len=cfg.max_target_len,
        fp16=cfg.fp16,
        seed=cfg.seed,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.train_bs,
        per_device_eval_batch_size=cfg.eval_bs,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.epochs,
        predict_with_generate=False,
        eval_strategy="no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        report_to="none",
    )

    runner.run(
        training_args=training_args,
        eval_each_epoch=cfg.eval_each_epoch,
        eval_stream_bs=cfg.eval_stream_bs,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    main()
