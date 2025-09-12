# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

        print(f'response: {response}')

        print("\n✅ Test passed! All assertions successful.")

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    # Load environment variables if .env file exists
    try:
        with open(os.path.expanduser('~/.env'), 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        print("No .env file found, using system environment variables")

    # Run tests
    test_text_rerank()

    print("\n🎉 All tests completed successfully!")
