#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import argparse
import pandas as pd
from datasets import Dataset
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank


def main():
    parser = argparse.ArgumentParser(description="Arguments for BM25+CE retriever.")
    parser.add_argument("--n_docs", type=int, required=True)
    parser.add_argument("--in_file", type=str, required=True, help="In file")
    parser.add_argument("--out_file", type=str, required=True, help="output file")
    cfg = parser.parse_args()

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    question_path = os.path.join(script_dir, cfg.in_file)
    question_df = pd.read_json(question_path, orient="index")
    question_df["close_time"] = pd.to_datetime(question_df["close_time"])
    questions = question_df["question"].to_dict()

    cc_news_path = os.path.join(script_dir, "cc_news")
    cc_news_df = Dataset.load_from_disk(cc_news_path).to_pandas()
    cc_news_df = cc_news_df.set_index("id")
    cc_news_df["date"] = pd.to_datetime(cc_news_df["date"])
    cc_news_df.index = cc_news_df.index.map(str)
    corpus = cc_news_df.to_dict(orient="index")

    model = BM25(
        hostname="http://localhost:9200",
        index_name="autocast",
        initialize=True,
    )
    retriever = EvaluateRetrieval(model)
    scores = retriever.retrieve(corpus, questions)

    scores = {
        question_id: {
            doc_id: score
            for doc_id, score in doc_scores.items()
            if cc_news_df.at[doc_id, "date"]
            <= question_df.at[question_id, "close_time"]
        }
        for question_id, doc_scores in scores.items()
    }
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-electra-base")
    reranker = Rerank(cross_encoder_model, batch_size=256)

    rerank_scores = reranker.rerank(
        corpus,
        questions,
        scores,
        top_k=cfg.n_docs,
    )

    research_material = {
        key: [int(doc_id) for doc_id in value.keys()]
        for key, value in rerank_scores.items()
    }

    out_path = os.path.join(script_dir, cfg.out_file)
    with open(out_path, "w") as outfile:
        json.dump(research_material, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
