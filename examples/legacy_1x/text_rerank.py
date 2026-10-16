# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""TextReRank document-ranking example."""

import os

from dashscope import TextReRank

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended)
MODEL = os.environ.get("DASHSCOPE_MODEL", "gte-rerank-v2")

QUERY = "哈尔滨在哪？"
DOCUMENTS = [
    "黑龙江离俄罗斯很近",
    "哈尔滨是中国黑龙江省的省会，位于中国东北",
]


if __name__ == "__main__":
    response = TextReRank.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        query=QUERY,
        documents=DOCUMENTS,
        return_documents=True,
        top_n=5,
        instruct="Retrieval document that can answer users query.",
    )
    for result in response.output.results:
        print(
            f"index={result['index']}, "
            f"score={result['relevance_score']}, "
            f"document={result.get('document', {}).get('text')}",
        )
