# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from dashscope import TextReRank

def test_text_rerank():
    """Test text rerank API with instruct parameter."""
    query = "哈尔滨在哪？"
    documents = [
        "黑龙江离俄罗斯很近",
        "哈尔滨是中国黑龙江省的省会，位于中国东北"
    ]

    try:
        response = TextReRank.call(
            model=os.getenv("MODEL_NAME"),
            query=query,
            documents=documents,
            return_documents=True,
            top_n=5,
            instruct="Retrieval document that can answer users query."
        )

        print(f'response:\n{response}')

    except Exception as e:
        raise

if __name__ == "__main__":
    test_text_rerank()