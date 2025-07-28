import asyncio
import struct
from dataclasses import dataclass
from typing import Optional
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

logger = init_logger(__name__)


@dataclass
class ReqState:
    req: EngineCoreRequest
    engines: list[str]
    engtokens: list[list[int]]


class MigrationFrontend:
    def __init__(self, vllm_config: VllmConfig, migration_instance_id: str):
        self._cfg = vllm_config
        self.migration_instance_id = migration_instance_id
        self._loop = asyncio.get_running_loop()
        self._inst_id = get_inst_id(vllm_config)
        self._naming_url = vllm_config.kv_transfer_config.get_from_extra_config(
            "naming_url", "badbad")
        self._naming_cli = connect_naming(self._inst_id, self._naming_url)
        self._pmgr = PeerManager(self._naming_cli, None)  # None means ALL
        self._pmgr.start(self._loop)
        self._enc = MsgpackEncoder()

    async def _do_migrate(self, msgbuf, retry: int, dst_migration_instance_id: Optional[str] = None) -> str:
        if dst_migration_instance_id is None:
            dst_migration_instance_id = self._pmgr.get_peer(exclude=self.migration_instance_id)
        if dst_migration_instance_id is None:
            raise RuntimeError("no instance")
        await self._rpc(dst_migration_instance_id, msgbuf, MIGRATE_TO_RESP, retry)
        logger.info(f"Migrate to {dst_migration_instance_id}")
        return dst_migration_instance_id

    async def migrate(self, req: ReqState, dst_migration_instance_id: Optional[str] = None):
        core_update_params(req.req, {SRC_INFO: self.migration_instance_id})
        msgbuf = bytearray.fromhex("00 00 00 00 00 00 00 00")
        struct.pack_into("=II", msgbuf, 0, MIGRATE_TO_REQ, 0)
        reqbufs = self._enc.encode_into(req.req, msgbuf, 8)
        assert len(reqbufs) == 1  # Need Support MsgpackEncoder.aux_buffers
        struct.pack_into("=I", msgbuf, 4, len(msgbuf) - 8)
        reqid = req.req.request_id
        peerid: Optional[str] = None
        for idx in range(2):
            try:
                peerid = await self._do_migrate(msgbuf, idx, dst_migration_instance_id=dst_migration_instance_id)
                break
            # pylint: disable=broad-except
            except Exception:
                logger.exception("migrate failed:reqid=%s try=%s", reqid, idx)
            await asyncio.sleep(0)
        if not peerid:
            logger.warning("migrate failed to finf any available peer")
            return
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
