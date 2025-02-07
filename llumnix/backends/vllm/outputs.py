from vllm.outputs import RequestOutput, RequestOutputFactory, EmbeddingRequestOutput

from llumnix.backends.vllm.sequence import SequenceGroupLlumnix


class LlumnixRequestOutputFactory(RequestOutputFactory):
    @staticmethod
    def create(seq_group: SequenceGroupLlumnix, use_cache: bool = False):
        # Determine the type based on a condition, for example:
        if hasattr(seq_group,
                   'embeddings') and seq_group.embeddings is not None:
            return EmbeddingRequestOutput.from_seq_group(seq_group), seq_group.server_info
        # pylint: disable=too-many-function-args
        return RequestOutput.from_seq_group(seq_group, use_cache), seq_group.server_info
