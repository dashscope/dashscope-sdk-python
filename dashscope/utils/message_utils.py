# Copyright (c) Alibaba, Inc. and its affiliates.

def merge_single_response(parsed_response, accumulated_data):
    """Merge a single response chunk with accumulated data."""
    # Handle output.text accumulation when choices is null
    if (parsed_response.output and
            hasattr(parsed_response.output, 'text') and
            parsed_response.output.text and
            (not parsed_response.output.choices or parsed_response.output.choices is None)):
        choice_idx = 0  # Use choice_idx 0 for output.text content to reuse existing structure
        if choice_idx not in accumulated_data:
            accumulated_data[choice_idx] = {
                'content': '',
                'reasoning_content': '',
                'tool_calls': [],
                'logprobs': {'content': []}
            }
        accumulated_data[choice_idx]['content'] += parsed_response.output.text
        parsed_response.output.text = accumulated_data[choice_idx]['content']
        return

    # Process each choice in the choices array
    if parsed_response.output and parsed_response.output.choices:
        for choice_enum_idx, choice in enumerate(parsed_response.output.choices):
            # Use choice.index if available, otherwise use enumerate index for compatibility
            try:
                choice_idx = choice.index if hasattr(choice, 'index') and 'index' in choice else choice_enum_idx
            except (KeyError, AttributeError):
                choice_idx = choice_enum_idx

            # Initialize accumulated data for this choice if not exists
            if choice_idx not in accumulated_data:
                accumulated_data[choice_idx] = {
                    'content': '',
                    'reasoning_content': '',
                    'tool_calls': [],
                    'logprobs': {'content': []}
                }

            if choice.message:
                # Handle content accumulation
                if 'content' in choice.message and choice.message.content:
                    current_content = choice.message.content
                    if current_content:
                        # Check if content is multimodal format (array with text fields)
                        if isinstance(current_content, list):
                            # Handle multimodal content (array format)
                            # Initialize accumulated content as array if not already
                            if not isinstance(accumulated_data[choice_idx]['content'], list):
                                accumulated_data[choice_idx]['content'] = []

                            # Ensure accumulated content list has enough elements
                            while len(accumulated_data[choice_idx]['content']) < len(current_content):
                                accumulated_data[choice_idx]['content'].append({'text': ''})

                            # Merge each content element
                            for content_idx, content_item in enumerate(current_content):
                                if isinstance(content_item, dict) and 'text' in content_item:
                                    if content_item['text']:
                                        # Accumulate text content
                                        accumulated_data[choice_idx]['content'][content_idx]['text'] += content_item['text']
                                        # Update the current response with accumulated content
                                        choice.message.content[content_idx]['text'] = accumulated_data[choice_idx]['content'][content_idx]['text']
                        else:
                            # Handle regular content (string format)
                            # Initialize accumulated content as string if not already
                            if isinstance(accumulated_data[choice_idx]['content'], list):
                                accumulated_data[choice_idx]['content'] = ''
                            accumulated_data[choice_idx]['content'] += current_content
                            choice.message.content = accumulated_data[choice_idx]['content']

                # Handle reasoning_content accumulation
                if 'reasoning_content' in choice.message:
                    current_reasoning_content = choice.message.reasoning_content
                    if current_reasoning_content:
                        accumulated_data[choice_idx]['reasoning_content'] += current_reasoning_content
                    # Always set the accumulated reasoning_content back, even if current is empty
                    choice.message.reasoning_content = accumulated_data[choice_idx]['reasoning_content']

                # Handle tool_calls accumulation
                if 'tool_calls' in choice.message and choice.message.tool_calls:
                    current_tool_calls = choice.message.tool_calls

                    # For each current tool call, accumulate its arguments
                    for current_call in current_tool_calls:
                        if isinstance(current_call, dict) and 'index' in current_call:
                            idx = current_call['index']

                            # Find existing accumulated call with same index
                            existing_call = None
                            for acc_call in accumulated_data[choice_idx]['tool_calls']:
                                if (isinstance(acc_call, dict) and
                                        acc_call.get('index') == idx):
                                    existing_call = acc_call
                                    break

                            if existing_call:
                                # Accumulate function fields from current call
                                if ('function' in current_call and
                                        current_call['function']):
                                    if 'function' not in existing_call:
                                        existing_call['function'] = {}

                                    # Accumulate function.name
                                    if 'name' in current_call['function']:
                                        if 'name' not in existing_call['function']:
                                            existing_call['function']['name'] = ''
                                        existing_call['function']['name'] += current_call['function']['name']

                                    # Accumulate function.arguments
                                    if 'arguments' in current_call['function']:
                                        if 'arguments' not in existing_call['function']:
                                            existing_call['function']['arguments'] = ''
                                        existing_call['function']['arguments'] += current_call['function']['arguments']

                                # Update other fields with latest values
                                existing_call.update({k: v for k, v in current_call.items()
                                                      if k != 'function' and v})
                                if 'function' in current_call and current_call['function']:
                                    existing_call['function'].update({k: v for k, v in current_call['function'].items()
                                                                      if k not in ['arguments', 'name'] and v})
                            else:
                                # Add new tool call
                                accumulated_data[choice_idx]['tool_calls'].append(dict(current_call))

                    # Update choice with accumulated tool_calls
                    choice.message.tool_calls = accumulated_data[choice_idx]['tool_calls']

            # Handle logprobs accumulation (only if logprobs exists)
            try:
                if ('logprobs' in choice and choice.logprobs and
                        isinstance(choice.logprobs, dict) and 'content' in choice.logprobs):
                    current_logprobs_content = choice.logprobs['content']
                    if current_logprobs_content and isinstance(current_logprobs_content, list):
                        # Initialize logprobs content if not exists
                        if 'logprobs' not in accumulated_data[choice_idx]:
                            accumulated_data[choice_idx]['logprobs'] = {'content': []}
                        elif 'content' not in accumulated_data[choice_idx]['logprobs']:
                            accumulated_data[choice_idx]['logprobs']['content'] = []

                        # Extend the accumulated logprobs content array
                        accumulated_data[choice_idx]['logprobs']['content'].extend(current_logprobs_content)
                        # Update choice with accumulated logprobs
                        choice.logprobs['content'] = accumulated_data[choice_idx]['logprobs']['content']
            except (KeyError, AttributeError, TypeError):
                # logprobs field might not exist or be in unexpected format, safely skip
                pass
