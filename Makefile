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

.PHONY: init
init:
	@git submodule update --init --recursive

.PHONY: install
install:
	@pip install -e .

.PHONY: lint
lint: check_pylint_installed
	@pylint --rcfile=.pylintrc -s n ./llumnix ./tests --exit-zero

.PHONY: test
test: check_pytest_installed
	@pytest -q -x --ignore=third_party/ --disable-warnings

#################### pygloo install for gloo migration backend begin ####################

BAZEL_CMD = bazel
PYGLOO_DIR = third_party/pygloo

.PHONY: pygloo
pygloo: init
	@./tools/pygloo_install.sh

##################### pygloo install for gloo migration backend end #####################

###################################### cupy begin #######################################

.PHONY: cupy-cuda
cupy-cuda:
	@./tools/cupy_cuda_install.sh

####################################### cupy end ########################################

##################################### pylint begin ######################################

PYLINT_VERSION = 2.12.2

.PHONY: check_pylint_installed
check_pylint_installed:
	@python3 -m pip show pylint > /dev/null 2>&1 || { \
		echo "pylint is not installed. Installing pylint $(PYLINT_VERSION)..."; \
		python3 -m pip install pylint==$(PYLINT_VERSION); }

###################################### pylint end #######################################

##################################### pytest begin ######################################

.PHONY: check_pytest_installed
check_pytest_installed:
	@python3 -m pip show pytest > /dev/null 2>&1 || { \
		echo "pytest is not installed. Installing pytest ..."; \
		python3 -m pip install pytest; }

	@python3 -m pip show pytest-asyncio > /dev/null 2>&1 || { \
		echo "pytest-asyncio is not installed. Installing pytest-asyncio ..."; \
		python3 -m pip install pytest-asyncio; }

	@python3 -m pip show pylint-pytest > /dev/null 2>&1 || { \
		echo "pylint-pytest is not installed. Installing pylint-pytest ..."; \
		python3 -m pip install pylint-pytest; }

###################################### pytest end #######################################
