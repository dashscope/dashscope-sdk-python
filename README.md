<h4 align="center">
    <p>
        <b>English</b>
    <p>
</h4>


</div>

# DashScope Python Library

## Installation
To install the DashScope Python SDK, simply run:
```shell
pip install dashscope
```

If you clone the code from github, you can install from  source by running:
```shell
pip install -e .
```

To use tokenizer in local mode without downloading any files, run:
```shell
pip install dashscope[tokenizer]
```


## QuickStart

You can use `Generation` api to call model qwen-turbo(通义千问).

```python
from http import HTTPStatus
import dashscope
from dashscope import Generation

dashscope.api_key = 'YOUR-DASHSCOPE-API-KEY'
responses = Generation.call(model=Generation.Models.qwen_turbo,
                            prompt='今天天气好吗？')

if responses.status_code == HTTPStatus.OK:
    print('Result is: %s' % responses.output)
else:
    print('Failed request_id: %s, status_code: %s, code: %s, message:%s' %
            (responses.request_id, responses.status_code, responses.code,
            responses.message))

```

## API Key Authentication

The SDK uses API key for authentication. Please refer to [official documentation for alibabacloud china](https://www.alibabacloud.com/help/en/model-studio/) and [official documentation for alibabacloud international](https://www.alibabacloud.com/help/en/model-studio/) regarding how to obtain your api-key.

### Using the API Key

1. Set the API key via code
```python
import dashscope

dashscope.api_key = 'YOUR-DASHSCOPE-API-KEY'
# Or specify the API key file path via code
# dashscope.api_key_file_path='~/.dashscope/api_key'

```

2. Set the API key via environment variables

a. Set the API key directly using the environment variable below

```shell
export DASHSCOPE_API_KEY='YOUR-DASHSCOPE-API-KEY'
```

b. Specify the API key file path via an environment variable

```shell
export DASHSCOPE_API_KEY_FILE_PATH='~/.dashscope/api_key'
```

3. Save the API key to a file
```python
from dashscope import save_api_key

save_api_key(api_key='YOUR-DASHSCOPE-API-KEY',
             api_key_file_path='api_key_file_location or (None, will save to default location "~/.dashscope/api_key"')

```


## Sample Code

`call` function provides  synchronous call, the function call will return when computation is done on the server side.

```python
from http import HTTPStatus
from dashscope import Generation
# export DASHSCOPE_API_KEY='YOUR-DASHSCOPE-API-KEY' in environment
def sync_dashscope_sample():
    responses = Generation.call(
        model=Generation.Models.qwen_turbo,
        prompt='Is the weather good today?')

    if responses.status_code == HTTPStatus.OK:
        print('Result is: %s'%responses.output)
    else:
        print('Code: %s, status_code: %s, code: %s, message: %s'%(responses.status_code,
                                                   responses.code,
                                                   responses.message))

if __name__ == '__main__':
    sync_dashscope_sample()
```

For requests with longer processing times, you can obtain partial results before the full output is generated. Set the **stream** parameter to **True**. In this case, the results will be returned in batches, and the current output mode is incremental (output will overwrite the previous content). When the output is in stream mode, the interface returns a generator, and you need to iterate through the generator to get the results. Each output contains partial data for streaming, and the last output contains the final generated result.

Example with simple streaming:
```python
from http import HTTPStatus
from dashscope import Generation

def sample_sync_call_stream():
    prompt_text = 'Give me a recipe using carrots, potatoes, and eggplants'
    response_generator = Generation.call(
        model=Generation.Models.qwen_turbo,
        prompt=prompt_text,
        stream=True,
        max_length=512,
        top_p=0.8)
    for resp in response_generator:  # Iterate through the streaming output results
        if resp.status_code == HTTPStatus.OK:
            print(resp.output)
        else:
            print('Request failed, message: %s'%resp.message)

if __name__ == '__main__':
    sample_sync_call_stream()

```
#### Stream with Messages
```python
from http import HTTPStatus
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role


def stream_with_messages():
    messages = [{'role': Role.SYSTEM, 'content': 'You are a helpful assistant.'},
                {'role': Role.USER, 'content': '如何做西红柿炖牛腩？'}]
    responses = Generation.call(
        Generation.Models.qwen_turbo,
        messages=messages,
        result_format='message',  # set the result to be "message" format.
        stream=True,
    )
    for response in responses:
       if response.status_code == HTTPStatus.OK:
           print(response)
       else:
           print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
               response.request_id, response.status_code,
               response.code, response.message
           ))

if __name__ == '__main__':
    stream_with_messages()

```
## Logging
To output Dashscope logs, you need to configure the logger.
```shell
export DASHSCOPE_LOGGING_LEVEL='info'

```

## Output
The output contains the following fields:
```
     request_id (str): The request id.
     status_code (int): HTTP status code, 200 indicates that the
         request was successful, others indicate an error。
     code (str): Error code if error occurs, otherwise empty str.
     message (str): Set to error message on error.
     output (Any): The request output.
     usage (Any): The request usage information.
```

## Error Handling
Currently, errors are thrown as exceptions.


## Contributing
Coming soon.


## License
This project is licensed under the Apache License (Version 2.0).
