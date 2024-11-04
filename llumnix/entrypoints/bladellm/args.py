# TODO[xinyi]: revise in bladellm repo

@dc.dataclass
class ServingArgs:
    # TODO[xinyi]: add lines in bladellm repo
    # llumnix scheduling mode
    enable_llumnix: bool = False
    llumnix_config: str = None

def add_args(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    parser.add_argument(
        "--enable_llumnix",
        type=bool,
        default=ServingArgs.enable_llumnix,
        help="enable llumnix to dispatch and migrate requests",
    )

    parser.add_argument("--llumnix-config",
        type=str,
        help="path to llumnix config file",
    )