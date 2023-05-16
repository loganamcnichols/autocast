# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from torch.utils.data import Dataset
import torch


class ForecastingQuestion(Dataset):
    def __init__(
        self,
        question: pd.Series,
        research_material: pd.DataFrame,
        label: int,
        n_context: int = 2,
        max_days: int = 128,
    ):
        """
        Example
        =======
        Consider the following question and assign a probability to each possible outcome.
        question: Before 1 October 2018, will either Russia or the United States announce
        that it is suspending its participation in or withdrawing from the Intermediate-Range
        Nuclear Forces Treaty? choices: 1. yes | 2. no
        Today's date is 2018-09-10.
        background: For more information on the Intermediate-Range Nuclear Forces (INF) Treaty
        see: U.S Department of State. Accusations of violations by both sides puts the future
        of the INF Treaty in question (Deutsche Welle).
        The follow articles may or may contain relevant information to the question:
        title: *title* body: *body*
        """
        self.research_material = research_material
        self.label = label
        self.n_context = n_context

        self.preamble = "Consider the following question and assign a probability to each possible outcome."
        # Format the question.
        self.question = "Question: " + question["question"]
        self.choices = question["choices"]
        if question["qtype"] == "mc":
            choices = question["choices"]
            formatted_choices = [f"{i+1}: {choice}" for i, choice in enumerate(choices)]
            choice_string = " | ".join(formatted_choices)
            self.question = f"{self.question} Choices: {choice_string}. "
        else:
            self.question = f"{self.question} choices: 1. yes | 2. no"
        self.background = f'Background: {question["background"]}'
        self.doc_preamble = "The follow articles may or may contain relevant information to the question:"
        # Get the question dates.
        start = question["publish_time"]
        end = question["close_time"]
        self.days = pd.date_range(start, end)[-max_days:]

    def __len__(self):
        return len(self.days)

    def __getitem__(self, index: int):
        date = self.days[index]
        date_str = date.strftime("%Y-%m-%d")
        date_str = f"Today's date is {date_str}."
        docs = self.research_material.loc[self.research_material["date"] <= date]
        doc_str = ""
        if docs.shape[0] > 0:
            docs = docs.sample(n=self.n_context, random_state=42, replace=True)
            publish_dates = docs["date"].dt.strftime("%Y-%m-%d")
            docs = (
                "Title: "
                + docs["title"]
                + " Published on: "
                + publish_dates
                + " Body: "
                + docs["text"]
            )
            doc_str = self.doc_preamble + " " + docs.str.cat(sep=" | ")
        example = (
            f"{date_str} {self.preamble} {self.question} {self.background} {doc_str}"
        )
        return {"target": self.label, "text": example}


class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        labels = []
        text = []
        for example in batch:
            labels.append(example["target"])
            text.append(example["text"])
        labels = torch.tensor(labels).view(-1, 1)
        passage_ids, passage_masks = self.encode_passages(text)
        return (labels, passage_ids, passage_masks)

    def encode_passages(self, batch_text_passages):
        passage_ids, passage_masks = [], []
        for text_passages in batch_text_passages:
            p = self.tokenizer.encode_plus(
                text_passages,
                max_length=self.text_maxlength,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            passage_ids.append(p["input_ids"][None])
            passage_masks.append(p["attention_mask"][None])

        passage_ids = torch.cat(passage_ids, dim=0)
        passage_masks = torch.cat(passage_masks, dim=0)
        return passage_ids.int(), passage_masks
