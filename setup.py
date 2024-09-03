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

from setuptools import setup, find_packages
import os
from typing import List

ROOT_DIR = os.path.dirname(__file__)

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

setup(
    name='llumnix',
    version='0.0.2',
    python_requires='>=3.8.1, <3.11',
    description='Efficient and easy multi-instance LLM serving',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Llumnix Team',
    url='https://github.com/AlibabaPAI/llumnix',
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=get_requirements(),
    platforms=["all"],
    classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          "License :: OSI Approved :: Apache Software License",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ],
)
