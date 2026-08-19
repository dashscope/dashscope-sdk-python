#!/bin/sh
#****************************************************************#
# ScriptName: query_local_reward_decorator.sh
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
  "agent_output": {
    "messages": [{"role": "user", "content": "I cannot help you with that"}]
  }
}' &

curl -X POST http://localhost:8000/api/v1 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer xxx" \
  -d '{
  "func_type": "reward",
  "ground_truth": "0.5",
  "agent_output": {
    "messages": [{"role": "user", "content": "I attack you with that"}]
  }
}' &

curl -X POST http://localhost:8000/api/v1 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer xxx" \
  -d '{
  "func_type": "reward",
  "ground_truth": "0.5",
  "agent_output": {
    "messages": [{"role": "user", "content": "I walk with you"}]
  }
}' &
