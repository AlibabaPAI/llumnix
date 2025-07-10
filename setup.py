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
import subprocess
from typing import List

import setuptools_scm
from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel

ROOT_DIR = os.path.dirname(__file__)
VERSION_SCHEME = setuptools_scm.version.guess_next_dev_version
LOCAL_SCHEME = setuptools_scm.version.get_local_node_and_date


# To auto generate version like 1.0.1.dev2+g1234abc {guessed_next_ver}.dev{distance}+g{git_hash}
# and write into blade_llm/version.py
def scm_version():
    return {
        'version_scheme': VERSION_SCHEME,
        'local_scheme': LOCAL_SCHEME,
        'write_to': 'llumnix/version.py',
        'write_to_template': '__version__ = {version!r}',
    }

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, 'requirements', *filepath)

def get_requirements(engine: str) -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path(f"requirements_{engine}.txt"), encoding="utf-8") as f:
        requirements = f.read().strip().split("\n")
    return requirements

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

class BuildWheelOverride(bdist_wheel):
    def run(self):
        subprocess.check_call(['make', 'proto'], cwd=ROOT_DIR)
        super().run()

setup(
    name='llumnix',
    use_scm_version=scm_version,
    python_requires='>=3.9.0, <=3.12.3',
    description='Efficient and easy multi-instance LLM serving',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Llumnix Team',
    url='https://github.com/AlibabaPAI/llumnix',
    license="Apache 2.0",
    packages=find_packages(),
    extras_require={
        'vllm': get_requirements('vllm'),
        'bladellm': get_requirements('bladellm'),
        'vllm_v1': get_requirements('vllm_v1'),
    },
    platforms=["all"],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    cmdclass={
        'bdist_wheel': BuildWheelOverride,
    }
)
