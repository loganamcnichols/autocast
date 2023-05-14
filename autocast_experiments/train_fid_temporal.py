# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import time
from datasets import Dataset
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from src.options import Options

import src.slurm
import src.util
import src.evaluation
from src.forecasting_data_multihead import ForecastingQuestion
import src.model

# from transformers import TransformerWrapper, Decoder
from transformers import GPT2Model
import pandas as pd


def train(
    model,
    encoder,
    optimizer,
    scheduler,
    all_params,
    step,
    train_dataset,
    eval_dataset,
    opt,
    day_collator,
    best_dev_em,
    checkpoint_path,
):
    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(
                Path(opt.checkpoint_dir) / opt.name
            )
        except:
            tb_logger = None
            logger.warning("Tensorboard is not available.")

    torch.manual_seed(opt.global_rank + opt.seed)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=2,
        collate_fn=lambda data: data,
    )

    model.train()
    for epoch in range(1, opt.epochs, 2):
        curr_loss, curr_loss_tf, curr_loss_mc, curr_loss_re = 0.0, 0.0, 0.0, 0.0
        em_tf, em_mc, em_re = [], [], []
        exactmatch = []
        crowd_em_tf, crowd_em_mc, crowd_em_re = [], [], []
        crowd_exactmatch = []
        my_preds_tf, my_preds_mc, my_preds_re = [], [], []
        time0 = time.time()

        for batch in train_dataloader:
            step += 1
            categories = []
            questions = []
            answers = []
            crowd_forecasts = []
            for category, question, answer, crowd_forecast in batch:
                question_encodings = encode_question(
                    encoder, question, opt, day_collator
                )
                crowd_forecast = crowd_forecast.to(device=question_encodings.device)

                questions.append(question_encodings)
                crowd_forecasts.append(crowd_forecast)
                answers.append(answer)
                categories.append(category)

            questions = pad_sequence(questions, batch_first=True)
            crowd_forecasts = pad_sequence(
                crowd_forecasts, batch_first=True, padding_value=-1
            )
            mask = (crowd_forecasts == -1).all(dim=2)
            categories = torch.tensor(categories, device=mask.device)
            answers = torch.tensor(answers, device=mask.device)
            seq_lengths = mask.sum(dim=1)

            hidden_state = model(questions, mask=mask)

            mc_logits = mc_classifier(hidden_state)[categories == 0]
            tf_logits = tf_classifier(hidden_state)[categories == 1]
            re_logits = regressor(hidden_state)[categories == 2]
            re_logits = re_logits.squeeze(-1)

            tf_probs = F.softmax(tf_logits, dim=-1)[..., 0]

            re_results = regressor(hidden_state).squeeze(-1)[categories == 2, ...]

            mc_labels = crowd_forecasts[categories == 0, :, :]
            tf_labels = crowd_forecasts[categories == 1, :, 0]
            re_labels = crowd_forecasts[categories == 2, :, 0]
            mc_mask = mask[categories == 0]
            tf_mask = mask[categories == 1]
            re_mask = mask[categories == 2]
            mc_mask_indi = mc_labels >= 0.0
            

            batch_loss, loss_mc, loss_re = (
                torch.tensor(0.0).cuda(),
                torch.tensor(0.0).cuda(),
                torch.tensor(0.0).cuda(),
            )
            size_tf, size_mc, size_re = (
                tf_mask.sum().item(),
                mc_mask.sum().item(),
                re_mask.sum().item(),
            )

            total_loss_mc = 0
            total_loss_tf = 0
            total_loss_num = 0
            if len(mc_labels) > 0:
                mc_logprobs = -nn.LogSoftmax(dim=2)(tf_logits)
                losses_mc = (mc_logprobs * mc_labels).sum(dim=2)
                total_loss_mc = (losses_mc * mc_mask).sum()
            if len(tf_labels) > 0:
                tf_logprobs = -nn.LogSoftmax(dim=2)(tf_logits)
                losses_true = tf_labels * tf_logprobs[:, :, 0]
                losses_false = (1 - tf_labels) * tf_logprobs[:, :, 1]
                losses_tf = losses_true + losses_false
                total_loss_tf = (losses_tf * tf_mask).sum()
            if len(re_labels) > 0:
                losses_num = nn.MSELoss(reduction="none")(re_results, re_labels)
                total_loss_num = (losses_num * re_mask).sum()

            train_loss = (total_loss_mc + total_loss_tf + total_loss_num) / mask.sum()  # TODO: re-weigh?

            train_loss.backward()

            seq_ends_indices_tf = seq_lengths[categories == 0].unsqueeze(-1)
            seq_ends_indices_mc = seq_lengths[categories == 1].unsqueeze(-1)
            seq_ends_indices_re = seq_lengths[categories == 2].unsqueeze(-1)
            seq_ends_expand = seq_ends_indices_mc.expand(
                -1, mc_labels.size()[-1]
            ).unsqueeze(1)
            

            true_labels_tf = answers[categories == 0]
            true_labels_mc = answers[categories == 1]
            true_labels_re = answers[categories == 2]

            if len(true_labels_tf) > 0:
                crowd_preds_tf = (
                    torch.gather(tf_labels.squeeze(-1), -1, seq_ends_indices_tf).view(
                        -1
                    )
                    > 0.5
                )
                preds_tf = (
                    torch.gather(tf_probs, -1, seq_ends_indices_tf).view(-1) > 0.5
                )
            else:
                crowd_preds_tf = torch.tensor([], device=true_labels_tf.device)
                preds_tf = torch.tensor([], device=true_labels_tf.device)

            if len(true_labels_mc) > 0:
                crowd_preds_mc = torch.argmax(
                    torch.gather(mc_labels, 1, seq_ends_expand).squeeze(1), dim=-1
                )
                preds_mc = torch.argmax(
                    torch.gather(mc_logits, 1, seq_ends_expand).squeeze(1), dim=-1
                )
            else:
                crowd_preds_mc = torch.tensor([], device=true_labels_mc.device)
                preds_mc = torch.tensor([], device=true_labels_mc.device)

            if len(true_labels_re) > 0:
                crowd_preds_re = torch.gather(re_labels, -1, seq_ends_indices_re).view(
                    -1
                )
                preds_re = torch.gather(re_results, -1, seq_ends_indices_re).view(-1)
            else:
                crowd_preds_re = torch.tensor([], device=true_labels_re.device)
                preds_re = torch.tensor([], device=true_labels_re.device)

            crowd_em_tf.extend(
                (true_labels_tf == crowd_preds_tf).detach().cpu().numpy()
            )
            em_tf.extend((true_labels_tf == preds_tf).detach().cpu().numpy())
            crowd_em_mc.extend(
                (true_labels_mc == crowd_preds_mc).detach().cpu().numpy()
            )
            em_mc.extend((true_labels_mc == preds_mc).detach().cpu().numpy())
            crowd_em_re.extend(
                -torch.abs(true_labels_re - crowd_preds_re).detach().cpu().numpy()
            )
            em_re.extend(-torch.abs(true_labels_re - preds_re).detach().cpu().numpy())

            my_preds_tf.extend(preds_tf.detach().cpu().numpy())
            my_preds_mc.extend(preds_mc.detach().cpu().numpy())
            my_preds_re.extend(preds_re.detach().cpu().numpy())

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(all_params, opt.clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if size_tf > 0:
                curr_loss_tf += src.util.average_main(batch_loss, opt).item()
            if size_mc > 0:
                curr_loss_mc += src.util.average_main(loss_mc, opt).item()
            if size_re > 0:
                curr_loss_re += src.util.average_main(loss_re, opt).item()


        dev_em, test_loss, crowd_em = evaluate(
            model,
            encoder,
            eval_dataset,
            day_collator,
            opt,
            epoch,
        )
        model.train()
        if opt.is_main:
            if dev_em > best_dev_em:
                best_dev_em = dev_em
            log = f"{step} / {opt.total_steps} | "
            log += f"train: {curr_loss / len(train_dataloader):.3f}; {curr_loss_tf / len(train_dataloader):.3f} / \
            {curr_loss_mc / len(train_dataloader):.3f} / {curr_loss_re / len(train_dataloader):.3f} | "
            log += f"test: {test_loss:.3f} | "
            log += f"evaluation: {100*dev_em:.2f} EM (crowd: {100*crowd_em:.2f} EM) | "
            log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
            logger.info(log)
            curr_loss = 0.0
            curr_loss_tf = 0.0
            curr_loss_mc = 0.0
            curr_loss_re = 0.0
            if tb_logger is not None:
                tb_logger.add_scalar("Evaluation", dev_em, step)
                tb_logger.add_scalar("Training", curr_loss, step)

        if not opt.epochs and step > opt.total_steps:
            return

    if opt.is_main:
        src.util.save(
            encoder,
            optimizer,
            scheduler,
            step,
            best_dev_em,
            opt,
            checkpoint_path,
            f"epoch-{epoch}-fidmodel",
        )
        src.util.save(
            model,
            optimizer,
            scheduler,
            step,
            best_dev_em,
            opt,
            checkpoint_path,
            f"epoch-{epoch}-gptmodel",
        )


def evaluate(model, encoder, dataset, day_collator, opt, epoch):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size * 4,
        drop_last=False,
        num_workers=2,
        collate_fn=lambda data: data,
    )
    model.eval()

    loss_fn_LSM = nn.LogSoftmax(dim=-1)
    loss_fn_re = nn.MSELoss(reduction="none")

    total_loss = 0.0
    em_tf, em_mc, em_re = [], [], []
    exactmatch = []
    crowd_em_tf, crowd_em_mc, crowd_em_re = [], [], []
    crowd_exactmatch = []
    my_preds_tf, my_preds_mc, my_preds_re = [], [], []
    time0 = time.time()
    device = torch.device("cpu")
    raw_logits = []
    with torch.no_grad():
        for batch in dataloader:
            step += 1
            categories = []
            questions = []
            answers = []
            crowd_forecasts = []
            for category, question, answer, crowd_forecast in batch:
                question_encodings = encode_question(
                    encoder, question, opt, day_collator
                )
                crowd_forecast = crowd_forecast.to(device=question_encodings.device)

                questions.append(question_encodings)
                crowd_forecasts.append(crowd_forecast)
                answers.append(answer)
                categories.append(category)

            questions = pad_sequence(questions, batch_first=True)
            crowd_forecasts = pad_sequence(
                crowd_forecasts, batch_first=True, padding_value=-1
            )
            mask = (crowd_forecasts == -1).all(dim=2)
            categories = torch.tensor(categories, device=mask.device)
            answers = torch.tensor(answers, device=mask.device)
            seq_lengths = mask.sum(dim=1)

            hidden_state = model(questions, mask=mask)

            mc_logits = mc_classifier(hidden_state)[categories == 0]
            tf_logits = tf_classifier(hidden_state)[categories == 1]
            re_logits = regressor(hidden_state)[categories == 2]
            re_logits = re_logits.squeeze(-1)

            tf_probs = F.softmax(tf_logits, dim=-1)[..., 0]

            re_results = regressor(hidden_state).squeeze(-1)[categories == 2, ...]

            mc_labels = crowd_forecasts[categories == 0, :, :]
            tf_labels = crowd_forecasts[categories == 1, :, 0]
            re_labels = crowd_forecasts[categories == 2, :, 0]
            mc_mask = mask[categories == 0]
            tf_mask = mask[categories == 1]
            re_mask = mask[categories == 2]
            mc_mask_indi = mc_labels >= 0.0
            

            batch_loss, loss_mc, loss_re = (
                torch.tensor(0.0).cuda(),
                torch.tensor(0.0).cuda(),
                torch.tensor(0.0).cuda(),
            )
            size_tf, size_mc, size_re = (
                tf_mask.sum().item(),
                mc_mask.sum().item(),
                re_mask.sum().item(),
            )

            total_loss_mc = 0
            total_loss_tf = 0
            total_loss_num = 0
            if len(mc_labels) > 0:
                mc_logprobs = -nn.LogSoftmax(dim=2)(tf_logits)
                losses_mc = (mc_logprobs * mc_labels).sum(dim=2)
                total_loss_mc = (losses_mc * mc_mask).sum()
            if len(tf_labels) > 0:
                tf_logprobs = -nn.LogSoftmax(dim=2)(tf_logits)
                losses_true = tf_labels * tf_logprobs[:, :, 0]
                losses_false = (1 - tf_labels) * tf_logprobs[:, :, 1]
                losses_tf = losses_true + losses_false
                total_loss_tf = (losses_tf * tf_mask).sum()
            if len(re_labels) > 0:
                losses_num = nn.MSELoss(reduction="none")(re_results, re_labels)
                total_loss_num = (losses_num * re_mask).sum()

            train_loss = (total_loss_mc + total_loss_tf + total_loss_num) / mask.sum()  # TODO: re-weigh?
            total_loss += train_loss.item()

            seq_ends_indices_tf = seq_ends[categories == 0].unsqueeze(-1)
            seq_ends_indices_mc = seq_ends[categories == 1].unsqueeze(-1)
            seq_ends_expand = seq_ends_indices_mc.expand(
                -1, mc_labels.size()[-1]
            ).unsqueeze(1)
            seq_ends_indices_re = seq_ends[categories == 2].unsqueeze(-1)

            true_labels_tf = answers[categories == 0]
            true_labels_mc = answers[categories == 1]
            true_labels_re = answers[categories == 2]

            if len(true_labels_tf) > 0:
                crowd_preds_tf = (
                    torch.gather(tf_labels.squeeze(-1), -1, seq_ends_indices_tf).view(
                        -1
                    )
                    > 0.5
                )
                preds_tf = (
                    torch.gather(tf_probs, -1, seq_ends_indices_tf).view(-1) > 0.5
                )
            else:
                crowd_preds_tf = torch.tensor([], device=true_labels_tf.device)
                preds_tf = torch.tensor([], device=true_labels_tf.device)

            if len(true_labels_mc) > 0:
                crowd_preds_mc = torch.argmax(
                    torch.gather(mc_labels, 1, seq_ends_expand).squeeze(1), dim=-1
                )
                preds_mc = torch.argmax(
                    torch.gather(mc_logits, 1, seq_ends_expand).squeeze(1), dim=-1
                )
            else:
                crowd_preds_mc = torch.tensor([], device=true_labels_mc.device)
                preds_mc = torch.tensor([], device=true_labels_mc.device)

            if len(true_labels_re) > 0:
                crowd_preds_re = torch.gather(re_labels, -1, seq_ends_indices_re).view(
                    -1
                )
                preds_re = torch.gather(re_results, -1, seq_ends_indices_re).view(-1)
            else:
                crowd_preds_re = torch.tensor([], device=true_labels_re.device)
                preds_re = torch.tensor([], device=true_labels_re.device)

            crowd_em_tf.extend(
                (true_labels_tf == crowd_preds_tf).detach().cpu().numpy()
            )
            em_tf.extend((true_labels_tf == preds_tf).detach().cpu().numpy())
            crowd_em_mc.extend(
                (true_labels_mc == crowd_preds_mc).detach().cpu().numpy()
            )
            em_mc.extend((true_labels_mc == preds_mc).detach().cpu().numpy())
            crowd_em_re.extend(
                -torch.abs(true_labels_re - crowd_preds_re).detach().cpu().numpy()
            )
            em_re.extend(-torch.abs(true_labels_re - preds_re).detach().cpu().numpy())

            my_preds_tf.extend(preds_tf.detach().cpu().numpy())
            my_preds_mc.extend(preds_mc.detach().cpu().numpy())
            my_preds_re.extend(preds_re.detach().cpu().numpy())

            tf_count, mc_count, re_count = 0, 0, 0
            seq_ends = seq_ends.detach().to(device).numpy() + 1
            tf_logits = tf_logits.detach().to(device).numpy()
            mc_logits = mc_logits.detach().to(device).numpy()
            re_results = re_results.detach().to(device).numpy()
            for i, cat in enumerate(categories):
                if cat == 0:
                    raw_logits.append(tf_logits[tf_count][: seq_ends[i]])
                    tf_count += 1
                elif cat == 1:
                    raw_logits.append(mc_logits[mc_count][: seq_ends[i]])
                    mc_count += 1
                elif cat == 2:
                    raw_logits.append(re_results[re_count][: seq_ends[i]])
                    re_count += 1

    # out-of-order overall stats
    crowd_exactmatch = crowd_em_tf + crowd_em_mc + crowd_em_re
    exactmatch = em_tf + em_mc + em_re

    if not checkpoint_path.exists():
        checkpoint_path.mkdir()
    outpath = checkpoint_path / f"results_epoch{epoch}.obj"
    with open(outpath, "wb") as f:
        pickle.dump(raw_logits, f)

    if len(em_tf) == 0:
        logger.info(f"EVAL:  For T/F: Predicted N/A")
    else:
        logger.info(
            f"EVAL:  For T/F: Predicted {em_tf.count(1)} Match {em_tf.count(0)} Wrong \
        ({my_preds_tf.count(1)} YES {my_preds_tf.count(0)} NO) | EM: {round(em_tf.count(1) / len(em_tf) * 100, 2)}"
        )
    if len(em_mc) == 0:
        logger.info(f"       For MC:  Predicted N/A")
    else:
        logger.info(
            f"       For MC:  Predicted {em_mc.count(1)} Match {em_mc.count(0)} Wrong | \
        EM: {round(em_mc.count(1) / len(em_mc) * 100, 2)}"
        )
    if len(em_re) == 0:
        logger.info(f"       For Reg: Predicted N/A")
    else:
        logger.info(f"       For Reg: Predicted Dist {np.mean(em_re)}")
    logger.info(f"{int(time.time() - time0)} sec")

    exactmatch, test_loss = src.util.weighted_average(
        np.mean(exactmatch) / 2, total_loss / len(dataloader), opt
    )
    return exactmatch, test_loss, np.mean(crowd_exactmatch) / 2


class ForecastingDataset:
    """
    Iterative predictions as sequence modeling
    where each token embeddings is replaced by
    hidden-state representation of daily news articles
    """

    def __init__(
        self,
        questions,
        crowd,
        schedule,
        corpus,
        max_days=128,
        n_context=None,
    ):
        self.schedule = schedule
        self.questions = questions.loc[schedule.keys(), :]
        self.crowd = crowd
        self.corpus = corpus
        self.max_days = max_days
        self.n_context = n_context

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question_data = self.questions.iloc[index, :]
        question_id = question_data.name
        schedule = self.schedule.get(question_id)
        crowd = self.crowd.get(question_id)
        schedule = pd.DataFrame(schedule).set_index("date")
        crowd = pd.DataFrame.from_dict(crowd, orient="index")
        schedule.index = pd.to_datetime(schedule.index)
        crowd.index = pd.to_datetime(crowd.index)
        active_dates = crowd.index.intersection(schedule.index)
        active_dates = active_dates.sort_values(ascending=False)[: self.max_days]
        schedule = schedule.loc[active_dates, :]
        crowd = crowd.loc[active_dates, :]
        research_materal = self.corpus.loc[schedule["doc_id"], :]
        answer = question_data.get("answer")
        qtype = question_data.get("qtype")
        if qtype == "mc":
            category = 0
            answer = bool(answer)
        elif qtype == "t/f":
            category = 1
            answer = int(answer)
        elif qtype == "num":
            category = 2

        crowd = torch.tensor(crowd.to_numpy())

        # Pad targets to be n x max_choice_len
        crowd_padded = torch.full((crowd.size(0), max_choices), -1.0)
        crowd_padded[: crowd.size(0), : crowd.size(1)] = crowd

        question = ForecastingQuestion(
            question_data,
            schedule,
            research_materal,
            self.n_context,
            max_choices,
        )
        return category, question, answer, crowd_padded


def encode_question(model, dataset, opt, collator, mode):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=16, drop_last=False, collate_fn=collator
    )

    outputs = []

    ### NO GRADIENTS ####
    if not opt.finetune_encoder:
        mode = "eval"
    ### NO GRADIENTS ####

    model.train(mode == "train")
    for batch in dataloader:
        (labels, context_ids, context_mask) = batch

        # TODO: we could pass in labels here too for additional training signal
        with torch.set_grad_enabled(mode == "train"):
            model_output = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda(),  # we use true labels for FiD hidden states
                output_hidden_states=True,
            )
        outputs.append(model_output.decoder_hidden_states[-1])

    outputs = torch.cat(outputs, dim=0)  # (n_examples, 1, hidden_size)
    outputs = outputs.view(outputs.shape[0], -1)  # (n_examples, hidden_size)

    if mode == "eval":
        outputs = outputs.detach()

    return outputs


def get_gpt(fid_hidden_size, gpt_hidden_size, opt, model_name="gpt2"):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    model = GPT2Model(config)
    model = model.cuda()

    input_embeddings = nn.Linear(fid_hidden_size, gpt_hidden_size).cuda()

    def gpt2_forward(self, X, mask):
        X = input_embeddings(X)
        # get last hidden state
        return model._call_impl(inputs_embeds=X, attention_mask=mask)[
            0
        ]  # last hidden state, (presents), (all hidden_states), (attentions)

    GPT2Model.__call__ = gpt2_forward

    tf_head = nn.Linear(gpt_hidden_size, 2)
    mc_head = nn.Linear(gpt_hidden_size, max_choices)
    regressor_head = nn.Sequential(nn.Linear(gpt_hidden_size, 1), nn.Sigmoid())

    tf_head = tf_head.cuda()
    mc_head = mc_head.cuda()
    regressor_head = regressor_head.cuda()

    return model, input_embeddings, tf_head, mc_head, regressor_head


tf_classifier, mc_classifier, regressor = None, None, None
max_choices = 12

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_forecaster_options()
    options.add_optim_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "run.log")

    logger = src.util.init_logger(opt.is_main, opt.is_distributed, log_path)

    model_name = "t5-" + opt.model_size
    model_class = src.model.FiDT5

    # load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    fid_collator = src.forecasting_data_multihead.Collator(
        opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength
    )

    if opt.model_path:
        checkpoint_path = Path(script_dir) / "checkpoint" / opt.model_path
    else:
        checkpoint_path = Path(script_dir) / "checkpoint" / "latest"

    if checkpoint_path.exists():
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = src.util.load(
            model_class, checkpoint_path, opt, reset_params=True
        )
        logger.info(f"Model loaded from {checkpoint_path}")
    else:
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    model.reset_head_to_identity()  # get hidden state output instead of lm_logits
    model = model.cuda()

    # Load data sources.
    ccnews_path = os.path.join(script_dir, "data/cc_news")
    corpus = Dataset.load_from_disk(ccnews_path).to_pandas().set_index("id")

    # Load training data.
    train_questions_path = os.path.join(script_dir, opt.train_questions)
    train_crowd_path = os.path.join(script_dir, opt.train_crowd)
    train_schedule_path = os.path.join(script_dir, opt.train_schedule)

    train_questions = pd.read_json(train_questions_path, orient="index")
    with open(train_crowd_path, "r") as datafile:
        train_crowd = json.load(datafile)
    with open(train_schedule_path, "r") as datafile:
        train_schedule = json.load(datafile)

    train_dataset = ForecastingDataset(
        train_questions,
        train_crowd,
        train_schedule,
        corpus,
        max_days=opt.max_seq_len,
        n_context=opt.n_context,
    )
    test_questions_path = os.path.join(script_dir, opt.test_questions)
    test_crowd_path = os.path.join(script_dir, opt.test_crowd)
    test_schedule_path = os.path.join(script_dir, opt.test_schedule)

    test_questions = pd.read_json(test_questions_path, orient="index")
    with open(test_crowd_path, "r") as datafile:
        test_crowd = json.load(datafile)
    with open(test_schedule_path, "r") as datafile:
        test_schedule = json.load(datafile)

    eval_dataset = ForecastingDataset(
        test_questions,
        test_crowd,
        test_schedule,
        corpus,
        max_days=opt.max_seq_len,
        n_context=opt.n_context,
    )

    gpt_model_name = "gpt2"
    if opt.model_size == "small":
        fid_hidden_size = 512
        gpt_hidden_size = 768
    elif opt.model_size == "base":
        fid_hidden_size = 768
        gpt_hidden_size = 1024
        gpt_model_name += "-medium"
    elif opt.model_size == "large":
        fid_hidden_size = 1024
        gpt_hidden_size = 1280
        gpt_model_name += "-large"
    elif opt.model_size == "3b":
        fid_hidden_size = 1024
        gpt_hidden_size = 1600
        gpt_model_name += "-xl"

    forecaster, input_embeddings, tf_classifier, mc_classifier, regressor = get_gpt(
        fid_hidden_size, gpt_hidden_size, opt, gpt_model_name
    )
    all_params = (
        list(model.parameters())
        + list(forecaster.parameters())
        + list(input_embeddings.parameters())
        + list(tf_classifier.parameters())
        + list(mc_classifier.parameters())
        + list(regressor.parameters())
    )
    optimizer, scheduler = src.util.set_optim(opt, model, all_params)

    logger.info(f"TRAIN EXAMPLE {len(train_dataset)}")
    logger.info(f"EVAL EXAMPLE {len(eval_dataset)}")
    logger.info("Start training")

    train(
        forecaster,
        model,
        optimizer,
        scheduler,
        all_params,
        step,
        train_dataset,
        eval_dataset,
        opt,
        fid_collator,
        best_dev_em,
        checkpoint_path,
    )
