#!/bin/sh
#****************************************************************#
# ScriptName: 1.sh
# Create Date: 2026-03-16 11:30
# Modify Date: 2026-03-16 11:30
# Function:
#***************************************************************#
curl -X POST https://dashscope.aliyuncs.com/api/v1/fine-tunes \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxx" \
  -d '{
	'model': 'qwen3-32b',
	'rollout': {
		'rollout_id': 'ro-50ada8b9-e027-4e43-9b6d-c0c875a8f06b',
		'cpu': 2,
		'memory_size': 4096,
		'disk_size': 10240,
		'concurrency': 2
	},
	'rewards': [{
		'reward_id': 'rw-4092eb99-8252-46a7-9919-3819df9484b1',
		'cpu': 2,
		'memory_size': 4096,
		'disk_size': 10240,
		'concurrency': 10
	}, {
		'reward_id': 'rw-38d37050-6c07-4b63-b014-fbb048c2a1c0',
		'cpu': 2,
		'memory_size': 4096,
		'disk_size': 10240,
		'concurrency': 5
	}],
	'training_file_ids': ['c5c39255-04d5-4d2f-9df5-c4db06107046'],
	'validation_file_ids': ['ec24290d-67bd-4378-93bf-9c23a27d4b79'],
	'hyper_parameters': {
		'batch_size': '256'
	},
	'training_type': 'reinforcement',
	'job_name': 'agentic-rl-3773091b',
	'model_name': ''
}'
