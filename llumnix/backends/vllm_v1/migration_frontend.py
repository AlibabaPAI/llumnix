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

import asyncio
import struct
from vllm.v1.hybrid_connector.engine_proxy import (EngineCoreRequest,
                            PlaceholderModule, MsgpackEncoder,
                            VllmConfig, core_update_params)
from vllm.v1.hybrid_connector.kvtbackend import PeerManager, get_inst_id
from vllm.v1.hybrid_connector.migration import MIGRATE_TO_REQ, SRC_INFO, MIGRATE_TO_RESP
try:
    import blade_kvt
    from blade_kvt.kv_transfer import connect_naming
except ImportError:
    blade_kvt = PlaceholderModule("blade_kvt")
    connect_naming = blade_kvt.placeholder_attr("connect_naming")

from llumnix.logging.logger import init_logger
from llumnix.utils import InstanceContext

logger = init_logger(__name__)


class MigrationFrontend:
    def __init__(self, vllm_config: VllmConfig, dp_rank: int):
        self._cfg = vllm_config

        assert vllm_config.kv_transfer_config is not None
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        kvt_id = get_inst_id(vllm_config)
        self.naming_instance_id = f"{kvt_id}|{dp_rank}|{tp_size}"

        self._loop = asyncio.get_running_loop()
        self._inst_id = get_inst_id(vllm_config)
        self._naming_url = vllm_config.kv_transfer_config.get_from_extra_config(
            "naming_url", "badbad")
        self._naming_cli = connect_naming(self._inst_id, self._naming_url)
        self._pmgr = PeerManager(self._naming_cli, None)  # None means ALL
        self._pmgr.start(self._loop)
        self._enc = MsgpackEncoder()

    async def _do_migrate(self, msgbuf, retry: int, dst_instance_context: InstanceContext) -> None:
        hint = (dst_instance_context.engine_host, dst_instance_context.kvt_engine_available_port)
        dst_naming_instance_id = self._pmgr.get_peer(hint=hint)
        if dst_naming_instance_id is None:
            raise RuntimeError(f"do_migrate failed: no target naming instance {dst_instance_context}")
        await self._rpc(dst_naming_instance_id, msgbuf, MIGRATE_TO_RESP, retry)

    async def migrate(self, req: EngineCoreRequest, dst_instance_context: InstanceContext):
        core_update_params(req, {SRC_INFO: self.naming_instance_id})
        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, MIGRATE_TO_REQ, 0)
        reqbufs = self._enc.encode_into(req, msgbuf, 8)
        assert len(reqbufs) == 1  # Need Support MsgpackEncoder.aux_buffers
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)
        reqid = req.request_id
        for idx in range(2):
            try:
                await self._do_migrate(msgbuf, idx, dst_instance_context=dst_instance_context)
                break
            # pylint: disable=broad-except
            except Exception:
                logger.exception("migration failed: reqid=%s try=%s", reqid, idx)
            await asyncio.sleep(0)
        return

    async def _rpc(self, peerid: str, msgbuf, resp: int, retry: int):
        pconn = await self._pmgr.acquire_conn(peerid, retry > 0)
        pconn[1].write(msgbuf)
        await pconn[1].drain()
        respbuf = await pconn[0].readexactly(4)
        (head, ) = struct.unpack("=I", respbuf)
        if head != resp:
            raise RuntimeError(f"invalid resp {head=}")
        self._pmgr.release_conn(peerid, pconn)
        return
