#!/bin/sh
#****************************************************************#
# ScriptName: query_reward.sh
# Create Date: 2026-03-16 11:44
# Modify Date: 2026-03-16 11:44
# Function:
#***************************************************************#
curl -X POST http://localhost:8000/api/v1 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer xxx" \
  -d '{
  "func_type": "reward",
  "ground_truth": "0.5",
  "request_metadata": null,
  "agent_output": {
    "messages": [{"role": "user", "content": "Test question"}],
    "rollout_extra": {
      "content": "This is a test content that exceeds 50 characters in length and should receive a reward score of 0.5"
    },
    "rollout_metrics": {"accuracy": 0.95},
    "reward_score": null
  },
  "rollout_id": "ro-ins-e8e92e94-b5c3-4546-b7b1-9832b8b948f2"
}'
