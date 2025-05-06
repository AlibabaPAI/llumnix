import signal
import subprocess
import sys
import os
import time
import ray

from llumnix.logging.logger import init_logger
from llumnix.entrypoints.setup import connect_to_ray_cluster

logger = init_logger("llumnix")

connect_to_ray_cluster()


@ray.remote
class EasLaunchActor:
    def __init__(self):
        self.process = None

    def launch_serve(self, command):
        # avoid "RuntimeError: No CUDA GPUs are available"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        logger.info("Launching command: {}".format(' '.join(command)))

        try:
            self.process = subprocess.Popen(
                command,
                preexec_fn=os.setsid
            )
            logger.info(f"Started serve launch command with PID {self.process.pid}")
        except Exception as e:
            logger.error(f"Subprocess error: {e}")
        return self.process.pid

    def stop(self):
        #TODO(tongchenghao): not work
        logger.info("Stopping subprocess...")
        if self.process and self.process.poll() is None:
            try:
                logger.info(f"Sent SIGTERM to process group {os.getpgid(self.process.pid)}")
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("Terminate timeout, using kill")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()


def find_gpu_node():
    gpu_nodes = ray.nodes()
    for node in gpu_nodes:
        if node["Alive"] and node["Resources"].get("GPU", 0) > 0:
            return node["NodeID"]
    return None


if __name__ == "__main__":

    # get serve args
    args = sys.argv[1:]
    cmd: list[str] = ["python", "-m", "llumnix.entrypoints.bladellm.serve"] + args

    gpu_node_id = find_gpu_node()
    logger.info(f"gpu_node_id: {gpu_node_id}")

    actor = EasLaunchActor.options(
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=gpu_node_id, soft=False
        )
    ).remote()
    actor.launch_serve.remote(command=cmd)

    try:
        while True:
            # avoid main process exit
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("catch KeyboardInterrupt, stop child process.")
        ray.get(actor.stop.remote())
        logger.info("child process stopped.")
        sys.exit(0)
