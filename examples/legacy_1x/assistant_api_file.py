# -*- coding: utf-8 -*-
"""Assistant file attachment example: associate a file with
an assistant.

Before running, set the ID of an already-uploaded file via the
DASHSCOPE_FILE_ID environment variable.
"""

import os

from dashscope import Assistants  # pylint: disable=no-name-in-module
from dashscope.assistants.files import Files

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), and the
# file ID via the DASHSCOPE_FILE_ID environment variable
FILE_ID = os.environ.get("DASHSCOPE_FILE_ID", "your-file-id")


if __name__ == "__main__":
    # Create an assistant
    assistant = Assistants.create(model="qwen3.7-plus")
    print(f"assistant_id={assistant.id}")

    # Associate the file with the assistant
    file = Files.create(assistant_id=assistant.id, file_id=FILE_ID)
    print(f"create file: {file}")

    # Query the associated file
    file = Files.get(assistant_id=assistant.id, file_id=FILE_ID)
    print(f"get file: {file}")
