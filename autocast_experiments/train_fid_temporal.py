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

            hidden_state = model(questions, mask=mask)

            mc_logits = mc_classifier(hidden_state)[categories == 0]
            tf_logits = tf_classifier(hidden_state)[categories == 1]
            # re_logits = regressor(hidden_state)[categories == 2]
            # re_logits = re_logits.squeeze(-1)

            mc_labels = crowd_forecasts[categories == 0, :, :]
            tf_labels = crowd_forecasts[categories == 1, :, 0]
            re_labels = crowd_forecasts[categories == 2, :, 0]
            mc_mask = mask[categories == 0]
            tf_mask = mask[categories == 1]
            # re_mask = mask[categories == 2]

            total_loss_mc = 0
            total_loss_tf = 0
            total_loss_num = 0
            if len(mc_labels) > 0:
                mc_logprobs = -nn.LogSoftmax(dim=2)(mc_logits)
                losses_mc = (mc_logprobs * mc_labels).sum(dim=2)
                total_loss_mc = (losses_mc * mc_mask).sum()
            if len(tf_labels) > 0:
                tf_logprobs = -nn.LogSoftmax(dim=2)(tf_logits)
                losses_true = tf_labels * tf_logprobs[:, :, 0]
                losses_false = (1 - tf_labels) * tf_logprobs[:, :, 1]
                losses_tf = losses_true + losses_false
                total_loss_tf = (losses_tf * tf_mask).sum()
            # if len(re_labels) > 0:
            #     losses_num = nn.MSELoss(reduction="none")(re_logits, re_labels)
            #     total_loss_num = (losses_num * re_mask).sum()

            train_loss = (total_loss_mc + total_loss_tf + total_loss_num) / mask.sum()  # TODO: re-weigh?

            train_loss.backward()

        model.train()

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

            hidden_state = model(questions, mask=mask)

            mc_logits = mc_classifier(hidden_state)[categories == 0]
            tf_logits = tf_classifier(hidden_state)[categories == 1]
            # re_logits = regressor(hidden_state)[categories == 2]
            # re_logits = re_logits.squeeze(-1)

            tf_probs = F.softmax(tf_logits, dim=-1)[..., 0]

            re_results = regressor(hidden_state).squeeze(-1)[categories == 2, ...]

            mc_labels = crowd_forecasts[categories == 0, :, :]
            tf_labels = crowd_forecasts[categories == 1, :, 0]
            # re_labels = crowd_forecasts[categories == 2, :, 0]
            mc_mask = mask[categories == 0]
            tf_mask = mask[categories == 1]
            # re_mask = mask[categories == 2]
            
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
            # if len(re_labels) > 0:
            #     losses_num = nn.MSELoss(reduction="none")(re_results, re_labels)
            #     total_loss_num = (losses_num * re_mask).sum()

            train_loss = (total_loss_mc + total_loss_tf + total_loss_num) / mask.sum()  # TODO: re-weigh?
            total_loss += train_loss.item()


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
