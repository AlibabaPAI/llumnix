# TODO[xinyi]: benchmark metrics
# TODO[xinyi]: revise in bladellm repo
# async def sink_resp(llm_resp: LLMResponse, ws: WebSocketResponse):
#     if llm_resp.is_ok():
#         stream = llm_resp.async_stream()
#         measure_handle = None
#         async for resp in stream:
#             logger.debug("Entrypoint got response: {} {}", type(resp), resp)
#             s = resp.model_dump_json()
#             await ws.send_str(s)
#             if 'llumnix' in sys.modules:
#                 _, measure_handle = llm_resp
#                 measure_single_resp(resp, measure_handle)
#         if 'llumnix' in sys.modules:
#                 measure_resp(measure_handle)
#     else:
#         if resp.error():
#             err_resp = GenerateStreamResponse(is_ok=False, is_finished=True, error_info=resp.error()).model_dump_json()
#             await ws.send_str(err_resp)

# def main():
#     parser = add_args()
#     args = parser.parse_args()
#     args = ServingArgs.from_cli_args(args)

#     # Check whether FP8 paged kvcache quant is appointed to use and could be imported under current arch.
#     # If not, fallback to non-quant kvcache.
#     if (
#         args.load_model_options.kv_cache_quant
#         in ['fp8_e5m2', 'fp8_e4m3', "mix_f852i4", "mix_f843i4", "mix_i8i4", "mix_i4i4"]
#         and not fp8_paged_enabled()
#     ):
#         logger.warning(
#             "Experimental feature FP8 KV-Cache could not be imported, architecture may be incompatible, fallback to non-quant KV-Cache."
#         )
#         args.load_model_options.kv_cache_quant = 'no_quant'

    # logger.remove()
    # logger.add(sys.stderr, level=args.log_level)
    # logger.info('================ Serving Arguments ================')
    # for k, v in args.__dict__.items():
    #     logger.info(f"{k:>20}: {v}")

    # # check port first
    # check_ports(args)

    # init_metric(args.serving_metric_options.metric_export_interval_sec, *args.metric_exporters)

    # loop = asyncio.get_event_loop()

    # if args.enable_llumnix:
    #     import llumnix
    #     engine_model_conf, llm_client, entrypoint_cls = setup_llumnix_api_server(args)
    # else:
    #     llm_engine = AsyncLLMEngine(args)
    #     engine_model_conf = llm_engine.model_conf
    #     llm_engine.start(loop)
    #     llm_client = llm_engine.get_client()
    #     entrypoint_cls = Entrypoint
    # try:
    #     generation_conf_processor = GenerationConfigProcessor(args.generation_configs, engine_model_conf)
    # except Exception:
    #     logger.exception('Failed to load generation config processor when create server.')
    #     generation_conf_processor = None

    # # start entrypoint server
    # web_app = entrypoint_cls(
    #     client=llm_client,
    #     model_conf=engine_model_conf,
    #     generation_conf_processor=generation_conf_processor,
    #     chat_template_path=args.load_model_options.chat_template,
    #     pp_enabled=args.pipeline_parallel_size > 1,
    # ).create_web_app()
    # logger.info(f"Entrypoint API ready at {args.host}:{args.port}")
    # web.run_app(web_app, host=args.host, port=args.port, loop=loop)