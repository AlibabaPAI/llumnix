from llumnix.common.config import Config as Cfg

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = Cfg()

# -----------------------------------------------------------------------------
# Prefill Decoding Disaggregation Config
# -----------------------------------------------------------------------------
_C.PDD_CONFIG = Cfg()
_C.PDD_CONFIG.ENABLE_PREFILL_DISAGGREATION = False
_C.PDD_CONFIG.PREFILL_INSTANCE_NUM = -1
_C.PDD_CONFIG.PREFILL_INSTANCE_TYPE = None