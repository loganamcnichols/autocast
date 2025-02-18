# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import time
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_
import transformers
from transformers import T5ForConditionalGeneration

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from src.options import Options
from src.forecasting_data_multihead import Collator

import src.slurm
import src.util
import src.evaluation
from src.forecasting_data_multihead import ForecastingQuestion
import src.model
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
sns.set()
# from transformers import TransformerWrapper, Decoder
from transformers import GPT2Model
import pandas as pd
from typing import Dict, Iterable, List, Tuple
import textwrap

DocID = int
QuestionID = str
QuestionAnswer = int
CrowdForecast = torch.Tensor


test: QuestionID = "sup"
def train(
    model,
    encoder,
    optimizer,
    scheduler,
    train_dataset,
    test_dataset,
    opt,
    day_collator,
    mc_classifier,
    all_params,
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
    plot_results = False
    # Initialize the SummaryWriter
    model.train()
    for epoch in range(opt.epochs):
        running_loss = 0
        for i, batch in enumerate(train_dataloader):
            time0 = time.time()
            questions = []
            answers = []
            crowd_forecasts = []
            for _, question, answer, crowd_forecast in batch:
                question_encodings = encode_question(
                    encoder, question, opt, day_collator
                )
                crowd_forecast = crowd_forecast.to(device=question_encodings.device)

                questions.append(question_encodings)
                crowd_forecasts.append(crowd_forecast)
                answers.append(answer)

            questions = pad_sequence(questions, batch_first=True)
            crowd_forecasts = pad_sequence(crowd_forecasts, batch_first=True)
            mask = ~(crowd_forecasts == 0).all(dim=2)
            answers = torch.tensor(answers, device=mask.device)

            hidden_state = model(questions, mask=mask)
            logits = mc_classifier(hidden_state)


            neg_logprobs = -nn.LogSoftmax(dim=2)(logits)
            losses_mc = (neg_logprobs * crowd_forecasts).sum(dim=2)
            train_loss = (losses_mc * mask).sum() / mask.sum()

            train_loss.backward()
            clip_grad_norm_(all_params, opt.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += train_loss.item()
            if i % 10 == 9:
                print("epoch:", epoch + 1, "batch:", i + 1)
                print("running loss:", running_loss / 10)
                print("time for batch", time.time() - time0)
                running_loss = 0.0

        if epoch + 1 == opt.epochs:
            plot_results=True
        evaluate(
            model,
            encoder,
            test_dataset,
            day_collator,
            opt,
            epoch,
            mc_classifier,
            plot_results
        )
        model.train()

        # if opt.is_main:
        #     src.util.save(
        #         encoder,
        #         optimizer,
        #         scheduler,
        #         step,
        #         best_dev_em,
        #         opt,
        #         checkpoint_path,
        #         f"epoch-{epoch}-fidmodel",
        #     )
        #     src.util.save(
        #         model,
        #         optimizer,
        #         scheduler,
        #         step,
        #         best_dev_em,
        #         opt,
        #         checkpoint_path,
        #         f"epoch-{epoch}-gptmodel",
        #     )


def evaluate(
    model, encoder, dataset, day_collator, opt, epoch, mc_classifier, plot_results=False
):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=2,
        collate_fn=lambda data: data,
    )
    model.eval()

    # Creating a PDF file to save all plots
    if plot_results:
        script_dir = script_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_file =os.path.join(script_dir, "evaluation_plots.pdf")
        pdf = PdfPages(pdf_file)

    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            qids = []
            questions = []
            answers = []
            crowd_forecasts = []
            for qid, question, answer, crowd_forecast in batch:
                question_encodings = encode_question(
                    encoder, question, opt, day_collator
                )
                crowd_forecast = crowd_forecast.to(device=question_encodings.device)
                qids.append(qid)
                questions.append(question_encodings)
                crowd_forecasts.append(crowd_forecast)
                answers.append(answer)

            questions = pad_sequence(questions, batch_first=True)
            crowd_forecasts = pad_sequence(crowd_forecasts, batch_first=True)
            mask = (crowd_forecasts != 0).any(dim=2)
            answers = torch.tensor(answers, device=mask.device)

            hidden_state = model(questions, mask=mask)
            logits = mc_classifier(hidden_state)

            # If plot_results is True, plot the required data
            if plot_results:
                # Select a random index
                random.seed(42)
                j = random.randint(0, hidden_state.shape[0] - 1)

                # Get the required data
                model_probs = nn.Softmax(dim=2)(logits)
                hidden_state_j = model_probs[j, :, answers[j]]
                crowd_forecast_j = crowd_forecasts[j, :, answers[j]]

                question_data = dataset.questions.loc[qids[j]]
                choices = question_data["choices"]

                # Ensure hidden_state_j and crowd_forecast_j are same length as dates
                dates = pd.date_range(question_data["publish_time"], question_data["close_time"])
                if len(dates) < hidden_state_j.shape[0]:
                    hidden_state_j = hidden_state_j[:len(dates)]
                    crowd_forecast_j = crowd_forecast_j[:len(dates)]
                if hidden_state_j.shape[0] < len(dates):
                    dates = dates[-hidden_state_j.shape[0]:]


                # Create the plot
                plt.figure(figsize=(10, 6))
                wrapped_question = textwrap.fill(question_data["question"], width=60)
                plt.suptitle(wrapped_question, fontsize=10)
                plt.title('Answer: ' + choices[question_data["answer"]], fontsize=8)
                plt.plot(dates, hidden_state_j.detach().cpu().numpy(), label='Model')
                plt.plot(dates, crowd_forecast_j.detach().cpu().numpy(), label='Crowd')
                
                plt.ylabel('Probability')
                plt.ylim([0, 1])
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()                
                # Save the plot to the pdf
                pdf.savefig()
                plt.close()

            neg_logprobs = -nn.LogSoftmax(dim=2)(logits)
            losses_mc = (neg_logprobs * crowd_forecasts).sum(dim=2)
            train_loss = (losses_mc * mask).sum() / mask.sum()

            total_loss += train_loss.item()

            if i % 10 == 9:
                print("epoch:", epoch, "batch:", i + 1)

    if plot_results:
        pdf.close()


class ForecastingDataset:
    """
    Iterative predictions as sequence modeling
    where each token embeddings is replaced by
    hidden-state representation of daily news articles
    """

    def __init__(
        self,
        questions: pd.DataFrame,
        corpus: pd.DataFrame,
        crowd: Dict[QuestionID, Iterable[float]],
        reading: Dict[QuestionID, List[DocID]],
        max_choice: int = 12,
        max_days: int = 128,
        n_context: int = 2,
    ):
        self.reading = reading
        self.questions = questions
        self.crowd = crowd
        self.corpus = corpus
        self.max_days = max_days
        self.n_context = n_context

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index: int) -> Tuple[QuestionID, ForecastingQuestion, QuestionAnswer, CrowdForecast]:
        question_data = self.questions.iloc[index]
        question_id: QuestionID = str(question_data.name)
        answer = question_data["answer"]
        doc_ids = self.reading[question_id]
        crowd = self.crowd[question_id]
        reading = self.corpus.loc[doc_ids, :]
        crowd = torch.tensor(crowd)
        crowd_padded: CrowdForecast = torch.zeros(crowd.size(0), max_choices)
        crowd_padded[: crowd.size(0), : crowd.size(1)] = crowd
        crowd_padded = crowd_padded[-self.max_days :]

        question = ForecastingQuestion(
            question_data,
            reading,
            answer,
            self.n_context,
            self.max_days,
        )
        return question_id, question, answer, crowd_padded


def encode_question(model, dataset, opt, collator, mode="train"):
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

    outputs = torch.cat(outputs, dim=0)
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

    mc_head = nn.Linear(gpt_hidden_size, max_choices)
    regressor_head = nn.Sequential(nn.Linear(gpt_hidden_size, 1), nn.Sigmoid())

    mc_head = mc_head.cuda()
    regressor_head = regressor_head.cuda()

    return model, input_embeddings, mc_head, regressor_head


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

    model_name: str = "t5-" + opt.model_size
    model_class = src.model.FiDT5

    # load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    fid_collator = Collator(opt.text_maxlength, tokenizer, opt.answer_maxlength)

    t5 = T5ForConditionalGeneration.from_pretrained(model_name)
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
    train_reading_path = os.path.join(script_dir, opt.train_schedule)
    train_questions = pd.read_json(train_questions_path, orient="index")
    with open(train_crowd_path, "r") as datafile:
        train_crowd = json.load(datafile)
    with open(train_reading_path, "r") as datafile:
        train_reading = json.load(datafile)
    train_dataset = ForecastingDataset(
        train_questions,
        corpus,
        train_crowd,
        train_reading,
        max_days=opt.max_seq_len,
        n_context=opt.n_context,
    )

    # Load test data.
    test_questions_path = os.path.join(script_dir, opt.test_questions)
    test_crowd_path = os.path.join(script_dir, opt.test_crowd)
    test_reading_path = os.path.join(script_dir, opt.test_schedule)
    test_questions = pd.read_json(test_questions_path, orient="index")
    with open(test_reading_path, "r") as datafile:
        test_reading = json.load(datafile)
    with open(test_crowd_path, "r") as datafile:
        test_crowd = json.load(datafile)
    eval_dataset = ForecastingDataset(
        test_questions,
        corpus,
        test_crowd,
        test_reading,
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
    else:
        fid_hidden_size = 1024
        gpt_hidden_size = 1600
        gpt_model_name += "-xl"

    forecaster, input_embeddings, mc_classifier, regressor = get_gpt(
        fid_hidden_size, gpt_hidden_size, opt, gpt_model_name
    )
    all_params = (
        list(model.parameters())
        + list(forecaster.parameters())
        + list(input_embeddings.parameters())
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
        train_dataset,
        eval_dataset,
        opt,
        fid_collator,
        mc_classifier,
        all_params,
    )
