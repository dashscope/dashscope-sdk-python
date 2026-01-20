# -*- coding: utf-8 -*-
import os

from dashscope import Assistants
from dashscope.assistants.files import Files

file_id = os.environ.get("DASHSCOPE_FILE_ID")

my_assistant = Assistants.create(model="qwen_plus")
print(f"创建assistant的结果为:{my_assistant}")

create_file = Files.create(assistant_id=my_assistant.id, file_id=file_id)
print(f"创建file的结果为:{create_file}")

get_file = Files.get(assistant_id=my_assistant.id, file_id=file_id)
print(f"获取file的结果为:{get_file}")
