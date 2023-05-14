# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class FiDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        question,
        research_schedule,
        research_material,
        n_context=None,
        question_prefix="question:",
        title_prefix="title:",
        passage_prefix="context:",
        choices_prefix="choices:",
        max_choices=12,
        cat=None,
    ):
        self.research_schedule = research_schedule.sort_index(ascending=False)
        self.research_material = research_material
        self.n_context = n_context

        if question["qtype"] == "mc":
            self.target = int(question["answer"])
        elif question["qtype"] == "t/f":
            self.target = max_choices + bool(question["answer"])
        elif question["qtype"] == "num":
            self.target = max_choices + 2
        # Format the question.
        self.question = question_prefix + " " + question["question"]
        self.choices = question["choices"]
        if question["qtype"] == "mc":
            choices = question["choices"]
            formatted_choices = [f"{i+1}: {choice}" for i, choice in enumerate(choices)]
            choice_string = " | ".join(formatted_choices)
            self.question = f"{self.question} {choices_prefix} {choice_string}."

        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.research_schedule.index.unique())

    def __getitem__(self, index):
        date = self.research_schedule.index.unique()[index]
        scores = self.research_schedule.loc[[date], :].set_index("doc_id")
        docs = scores.join(self.research_material)
        docs = docs.sample(n=self.n_context, replace=True)
        passages = "title: " + docs["title"] + " context: " + docs["text"]
        return {
            "question": self.question,
            "passages": passages,
            "target": self.target,
        }


class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        labels = []
        text_passages = []
        for example in batch:
            labels.append(example["target"])
            passages = example["question"] + " " + example["passages"]
            text_passages.append(passages.to_list())
        labels = torch.tensor(labels).view(-1, 1)
        passage_ids, passage_masks = self.encode_passages(text_passages)
        return (labels, passage_ids, passage_masks)

    def encode_passages(self, batch_text_passages):
        passage_ids, passage_masks = [], []
        for text_passages in batch_text_passages:
            p = self.tokenizer.batch_encode_plus(
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
