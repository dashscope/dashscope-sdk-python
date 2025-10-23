# Copyright (c) Alibaba, Inc. and its affiliates.

def merge_single_response(parsed_response, accumulated_data, n=1):
    """Merge a single response chunk with accumulated data.

    Args:
        parsed_response: The response chunk to merge
        accumulated_data: Dictionary storing accumulated data for each choice
        n: Number of expected choices (default 1)

    Returns:
        bool: True if this response should be yielded, False if filtered
    """
    # Check if all choices have been sent (for n > 1 case)
    if n > 1 and accumulated_data:
        all_sent = any(data.get('all_choices_sent', False)
                       for data in accumulated_data.values())
        if all_sent:
            return False

    # Handle output.text accumulation when choices is null
    if (parsed_response.output and
            hasattr(parsed_response.output, 'text') and
            (not parsed_response.output.choices or parsed_response.output.choices is None)):
        choice_idx = 0
        if choice_idx not in accumulated_data:
            accumulated_data[choice_idx] = {
                'content': '',
                'reasoning_content': '',
                'tool_calls': [],
                'logprobs': {'content': []},
                'finished': False,
                'finish_reason': None,
                'all_choices_sent': False,
                'role': None
            }
        # Accumulate text if not empty
        if parsed_response.output.text:
            accumulated_data[choice_idx]['content'] += parsed_response.output.text
        # Always set accumulated content back to response
        parsed_response.output.text = accumulated_data[choice_idx]['content']
        return True

    # Process each choice in the choices array
    if parsed_response.output and parsed_response.output.choices:
        choices = parsed_response.output.choices

        # Filter out empty choices array
        if not choices:
            return False

        for choice_enum_idx, choice in enumerate(choices):
            # Use choice.index if available, otherwise use enumerate index
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
                    'logprobs': {'content': []},
                    'finished': False,
                    'finish_reason': None,
                    'all_choices_sent': False,
                    'role': None
                }

            # Handle message field - create if null
            if not choice.message:
                # Create message object with accumulated data
                choice.message = {
                    'role': accumulated_data[choice_idx]['role'] if accumulated_data[choice_idx]['role'] else 'assistant',
                    'content': accumulated_data[choice_idx]['content']
                }
                if accumulated_data[choice_idx]['reasoning_content']:
                    choice.message['reasoning_content'] = accumulated_data[choice_idx]['reasoning_content']
                if accumulated_data[choice_idx]['tool_calls']:
                    choice.message['tool_calls'] = accumulated_data[choice_idx]['tool_calls']
            else:
                # Save role if present
                if hasattr(choice.message, 'role') and choice.message.role:
                    accumulated_data[choice_idx]['role'] = choice.message.role

                # Handle content accumulation
                if 'content' in choice.message:
                    current_content = choice.message.content
                    if current_content:
                        # Check if content is multimodal format
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
                            for content_idx in range(len(accumulated_data[choice_idx]['content'])):
                                if content_idx < len(choice.message.content):
                                    choice.message.content[content_idx]['text'] = accumulated_data[choice_idx]['content'][content_idx]['text']
                        else:
                            # Handle regular content (string format)
                            # Initialize accumulated content as string
                            if isinstance(accumulated_data[choice_idx]['content'], list):
                                accumulated_data[choice_idx]['content'] = ''
                            # Accumulate content if not empty
                            accumulated_data[choice_idx]['content'] += current_content
                    # Always set accumulated content back to response
                    if not isinstance(accumulated_data[choice_idx]['content'], list):
                        choice.message.content = accumulated_data[choice_idx]['content']
                    else:
                        # For multimodal content, ensure message.content
                        # exists
                        if not isinstance(choice.message.content, list):
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

                # Restore role if we have it
                if accumulated_data[choice_idx]['role'] and (not hasattr(choice.message, 'role') or not choice.message.role):
                    choice.message.role = accumulated_data[choice_idx]['role']

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
            except (KeyError, AttributeError, TypeError):
                # logprobs field might not exist or be in unexpected format, safely skip
                pass

            # Always set accumulated logprobs if we have any
            if (accumulated_data[choice_idx]['logprobs']['content'] and
                    hasattr(choice, 'logprobs') and choice.logprobs):
                choice.logprobs['content'] = accumulated_data[choice_idx][
                    'logprobs']['content']

            # Handle finish_reason for n > 1 case
            if (n > 1 and hasattr(choice, 'finish_reason') and
                    choice.finish_reason and
                    choice.finish_reason != 'null'):
                accumulated_data[choice_idx]['finish_reason'] = \
                    choice.finish_reason
                accumulated_data[choice_idx]['finished'] = True

        # Check if all choices are finished when n > 1
        if n > 1:
            finished_count = sum(1 for data in accumulated_data.values()
                                 if data.get('finished', False))

            # If not all choices finished, hide finish_reason
            if finished_count < n:
                for choice in choices:
                    if (hasattr(choice, 'finish_reason') and
                            choice.finish_reason and
                            choice.finish_reason != 'null'):
                        choice.finish_reason = 'null'
            else:
                # All choices finished, mark as sent first
                for data in accumulated_data.values():
                    data['all_choices_sent'] = True

                # Return final result with all choices
                all_choices = []
                for choice_idx, data in accumulated_data.items():
                    # Create a new choice object
                    final_choice_dict = {
                        'index': choice_idx,
                        'finish_reason': data['finish_reason']
                    }

                    # Create message
                    message_dict = {
                        'role': data['role'] if data['role'] else 'assistant'
                    }
                    if data['content']:
                        message_dict['content'] = (
                            data['content'] if isinstance(data['content'], str)
                            else data['content'])
                    if data['reasoning_content']:
                        message_dict['reasoning_content'] = data['reasoning_content']
                    if data['tool_calls']:
                        message_dict['tool_calls'] = data['tool_calls']

                    final_choice_dict['message'] = message_dict

                    # Add logprobs if present
                    if data['logprobs']['content']:
                        final_choice_dict['logprobs'] = {
                            'content': data['logprobs']['content']
                        }

                    all_choices.append(final_choice_dict)

                # Update output choices with all accumulated choices
                parsed_response.output.choices = all_choices

    return True
