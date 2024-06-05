ssh root@172.23.75.192 'docker exec -w /workspace/llumnix llumnix-artifact conda run -n artifact pip install -e .'
ssh root@172.23.75.193 'docker exec -w /workspace/llumnix llumnix-artifact conda run -n artifact pip install -e .'
ssh root@172.23.75.194 'docker exec -w /workspace/llumnix llumnix-artifact conda run -n artifact pip install -e .'
ssh root@172.23.75.195 'docker exec -w /workspace/llumnix llumnix-artifact conda run -n artifact pip install -e .'