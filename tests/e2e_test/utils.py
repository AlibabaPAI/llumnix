# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil

from llumnix.utils import random_uuid

def to_markdown_table(data):
    headers = data[0]
    rows = data[1:]

    col_widths = [max(len(str(item)) for item in col) for col in zip(*data)]

    header_row = " | ".join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(headers))
    separator_row = " | ".join('-' * col_widths[i] for i in range(len(headers)))

    data_rows = []
    for row in rows:
        data_row = " | ".join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(row))
        data_rows.append(data_row)

    table = f"{header_row}\n{separator_row}\n" + "\n".join(data_rows) + "\n\n"
    return table

def backup_instance_log():
    dest_directory = os.path.expanduser(f'/mnt/error_log/{random_uuid()}/')
    os.makedirs(dest_directory, exist_ok=True)

    src_directory = os.getcwd()

    for filename in os.listdir(src_directory):
        if filename.startswith("instance_"):
            src_file_path = os.path.join(src_directory, filename)
            shutil.copy(src_file_path, dest_directory)
            print(f"Copied instance log: {src_file_path} to {dest_directory}")

        if filename.startswith("bench_"):
            src_file_path = os.path.join(src_directory, filename)
            shutil.copy(src_file_path, dest_directory)
            print(f"Copied bench log: {src_file_path} to {dest_directory}")
