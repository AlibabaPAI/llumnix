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
	@git submodule update --init --recursive --remote

.PHONY: vllm_install
vllm_install:
	@pip install -e .[vllm]

.PHONY: bladellm_install
bladellm_install:
	@pip install -e .[bladellm]
	@make proto

.PHONY: vllm_v1_install
vllm_v1_install:
	@pip install -e .[vllm_v1]

.PHONY: lint
lint: check_pylint_installed check_pytest_installed
	@err=0; \
	pylint --rcfile=.pylintrc -s n --jobs=128 ./llumnix setup.py --ignore=llumnix/backends/bladellm/proto || err=1; \
	pylint --rcfile=.pylintrc \
			--disable=protected-access,super-init-not-called,unused-argument,redefined-outer-name,invalid-name \
			-s n --jobs=128 ./tests/e2e_test ./tests/unit_test ./tests/conftest.py || err=1; \
	if [ "$$err" -ne "0" ]; then exit 1; fi

.PHONY: clean
clean: proto-clean

###################################### proto begin ######################################

.PHONY: proto
proto:
	@find . -type d -name "proto" -not -path "./build/*" | while read dir; do \
	    dir_base=$$(dirname $$dir); \
	    find $$dir -name "*.proto" | while read proto_file; do \
	        echo "Compiling $$proto_file"; \
	        PYTHONWARNINGS="ignore::DeprecationWarning" python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. $$proto_file; \
	    done; \
	done;

.PHONY: proto-clean
proto-clean:
	@find . -name "*_pb2_grpc.py" | xargs rm -f
	@find . -name "*_pb2.py" | xargs rm -f

####################################### proto end #######################################

###################################### test begin #######################################

.PHONY: vllm_test
vllm_test: check_pytest_installed vllm_unit_test vllm_offline_test vllm_correctness_test vllm_bench_test vllm_migration_test vllm_register_service_test

.PHONY: bladellm_test
bladellm_test: check_pytest_installed bladellm_correctness_test bladellm_bench_test bladellm_migration_test bladellm_register_service_test bladellm_server_test

.PHONY: bladellm_unit_test
bladellm_unit_test: check_pytest_installed
	@pytest -v --timer-top-n=999 --ignore=third_party --disable-warnings ./tests/unit_test/**/bladellm/

.PHONY: vllm_unit_test
vllm_unit_test: check_pytest_installed
	@pytest -v --timer-top-n=999 --ignore=third_party --ignore-glob="tests/**/bladellm" --disable-warnings ./tests/unit_test/

.PHONY: vllm_offline_test
vllm_offline_test:
	@python examples/offline_inference.py

# TODO(KuilongCui): add bladellm offine inference example

.PHONY: vllm_correctness_test
vllm_correctness_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_vLLM and not _v1 or not engine_' --tb=long ./tests/e2e_test/test_correctness.py

.PHONY: vllm_v1_correctness_test
vllm_v1_correctness_test: check_pytest_installed
	@NCCL_SOCKET_IFNAME=eth0 pytest -v -x -s -k 'engine_vLLM_v1 or not engine_' --tb=long ./tests/e2e_test/test_correctness.py

.PHONY: bladellm_correctness_test
bladellm_correctness_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_BladeLLM or not engine_' --tb=long ./tests/e2e_test/test_correctness.py

.PHONY: vllm_trace_test
vllm_trace_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_vLLM and not _v1 or not engine_' --tb=long ./tests/e2e_test/test_trace_request.py

.PHONY: bladellm_trace_test
bladellm_trace_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_BladeLLM or not engine_' --tb=long ./tests/e2e_test/test_trace_request.py

.PHONY: vllm_bench_test
vllm_bench_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_vLLM and not _v1 or not engine_' --tb=long ./tests/e2e_test/test_bench.py

.PHONY: bladellm_bench_test
bladellm_bench_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_BladeLLM or not engine_' --tb=long ./tests/e2e_test/test_bench.py

.PHONY: vllm_migration_test
vllm_migration_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_vLLM and not _v1 or not engine_' --tb=long ./tests/e2e_test/test_migration.py

.PHONY: bladellm_migration_test
bladellm_migration_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_BladeLLM or not engine_' --tb=long ./tests/e2e_test/test_migration.py

.PHONY: vllm_register_service_test
vllm_register_service_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_vLLM and not _v1 or not engine_' --tb=long ./tests/e2e_test/test_register_service.py

.PHONY: vllm_v1_register_service_test
vllm_v1_register_service_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_vLLM_v1 or not engine_' --tb=long ./tests/e2e_test/test_register_service.py

.PHONY: bladellm_register_service_test
bladellm_register_service_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_BladeLLM or not engine_' --tb=long ./tests/e2e_test/test_register_service.py

.PHONY: bladellm_server_test
bladellm_server_test: check_pytest_installed
	@pytest -v -x -s -k 'engine_BladeLLM or not engine_' --tb=long ./tests/e2e_test/test_server.py

####################################### test end ########################################

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
	@python3 -c "import pylint; assert pylint.__version__ == '$(PYLINT_VERSION)'" 2>/dev/null || { \
		echo "pylint is not installed or version does not match $(PYLINT_VERSION). Installing..."; \
		python3 -m pip install --force-reinstall pylint==$(PYLINT_VERSION); }

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

	@python3 -m pip show pytest-timeout > /dev/null 2>&1 || { \
		echo "pytest-timeout is not installed. Installing pytest-timeout ..."; \
		python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ pytest-timeout; }

	@python3 -m pip show pytest-timer > /dev/null 2>&1 || { \
		echo "pytest-timer is not installed. Installing pytest-timer ..."; \
		python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ pytest-timer; }

###################################### pytest end #######################################
