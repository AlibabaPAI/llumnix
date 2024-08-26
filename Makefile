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
install: cupy
	@pip install -e .

.PHONY: lint
lint: check_pylint_installed
	@pylint --rcfile=.pylintrc -s n ./llumnix

	@pylint -s n --disable=all \
			--enable=trailing-whitespace,unused-variable,wrong-import-order,missing-final-newline,line-too-long,\
			unused-import,singleton-comparison,unnecessary-comprehension ./tests

.PHONY: test
test:
	@pytest -q -x --ignore=third_party/ --disable-warnings

#################### pygloo install for gloo migration backend begin ####################

BAZEL_CMD = bazel
PYGLOO_DIR = third_party/pygloo

.PHONY: pygloo
pygloo: init
	@./tools/pygloo_install.sh

##################### pygloo install for gloo migration backend end #####################

###################################### cupy begin #######################################

.PHONY: cupy
cupy:
	@./tools/cupy_install.sh

####################################### cupy end ########################################

##################################### pylint begin ######################################

PYLINT_VERSION = 2.12.2

.PHONY: check_pylint_installed
check_pylint_installed:
	@command -v pylint >/dev/null 2>&1 || { \
		echo "pylint is not installed. Installing pylint $(PYLINT_VERSION)..."; \
		python3 -m pip install pylint==$(PYLINT_VERSION); }

###################################### pylint end #######################################
