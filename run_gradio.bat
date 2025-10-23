git pull
@echo off
REM Get the directory of the currently executed script
set SCRIPT_DIR=%~dp0

REM Navigate to the script directory
cd /d "%SCRIPT_DIR%"

REM Activate the virtual environment
CALL venv\Scripts\activate

REM Install/update dependencies
pip install --no-cache-dir -r requirements.txt

REM Run the conversion script
REM Launches Gradio WebUI with enhanced bit depth and seamless controls
python run_gradio.py

pause

[notice] A new release of pip is available: 23.0.1 -> 25.2
[notice] To update, run: python.exe -m pip install --upgrade pip
xFormers not available
xFormers not available
Select the encoder for the Gradio app:
1. vits
2. vitb
3. vitl
Enter the number corresponding to the encoder (1-3) or press Enter to use default (3): 3
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\uvicorn\protocols\http\h11_impl.py", line 403, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\uvicorn\middleware\proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\fastapi\applications.py", line 1133, in __call__
    await super().__call__(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\applications.py", line 113, in __call__
    await self.middleware_stack(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\middleware\errors.py", line 186, in __call__
    raise exc
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\middleware\errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\gradio\brotli_middleware.py", line 74, in __call__
    return await self.app(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\gradio\route_utils.py", line 882, in __call__
    await self.app(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\middleware\exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\routing.py", line 736, in app
    await route.handle(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\routing.py", line 290, in handle
    await self.app(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\fastapi\routing.py", line 123, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\fastapi\routing.py", line 110, in app
    await response(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\responses.py", line 369, in __call__
    await self._handle_simple(send, send_header_only, send_pathsend)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\responses.py", line 400, in _handle_simple
    await send({"type": "http.response.body", "body": chunk, "more_body": more_body})
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 39, in sender
    await send(message)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 39, in sender
    await send(message)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\middleware\errors.py", line 161, in _send
    await send(message)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\uvicorn\protocols\http\h11_impl.py", line 500, in send
    output = self.conn.send(event=h11.Data(data=data))
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\h11\_connection.py", line 538, in send
    data_list = self.send_with_data_passthrough(event)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\h11\_connection.py", line 571, in send_with_data_passthrough
    writer(event, data_list.append)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\h11\_writers.py", line 65, in __call__
    self.send_data(event.data, write)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\h11\_writers.py", line 91, in send_data
    raise LocalProtocolError("Too much data for declared Content-Length")
h11._util.LocalProtocolError: Too much data for declared Content-Length
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\uvicorn\protocols\http\h11_impl.py", line 403, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\uvicorn\middleware\proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\fastapi\applications.py", line 1133, in __call__
    await super().__call__(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\applications.py", line 113, in __call__
    await self.middleware_stack(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\middleware\errors.py", line 186, in __call__
    raise exc
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\middleware\errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\gradio\brotli_middleware.py", line 74, in __call__
    return await self.app(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\gradio\route_utils.py", line 882, in __call__
    await self.app(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\middleware\exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\routing.py", line 736, in app
    await route.handle(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\routing.py", line 290, in handle
    await self.app(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\fastapi\routing.py", line 123, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\fastapi\routing.py", line 110, in app
    await response(scope, receive, send)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\responses.py", line 369, in __call__
    await self._handle_simple(send, send_header_only, send_pathsend)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\responses.py", line 400, in _handle_simple
    await send({"type": "http.response.body", "body": chunk, "more_body": more_body})
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 39, in sender
    await send(message)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\_exception_handler.py", line 39, in sender
    await send(message)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\starlette\middleware\errors.py", line 161, in _send
    await send(message)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\uvicorn\protocols\http\h11_impl.py", line 507, in send
    output = self.conn.send(event=h11.EndOfMessage())
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\h11\_connection.py", line 538, in send
    data_list = self.send_with_data_passthrough(event)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\h11\_connection.py", line 571, in send_with_data_passthrough
    writer(event, data_list.append)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\h11\_writers.py", line 67, in __call__
    self.send_eom(event.headers, write)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\h11\_writers.py", line 96, in send_eom
    raise LocalProtocolError("Too little data for declared Content-Length")
h11._util.LocalProtocolError: Too little data for declared Content-Length
Traceback (most recent call last):
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\gradio\queueing.py", line 759, in process_events
    response = await route_utils.call_process_api(
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\gradio\route_utils.py", line 354, in call_process_api
    output = await app.get_blocks().process_api(
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\gradio\blocks.py", line 2116, in process_api
    result = await self.call_function(
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\gradio\blocks.py", line 1623, in call_function
    prediction = await anyio.to_thread.run_sync(  # type: ignore
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\anyio\to_thread.py", line 56, in run_sync
    return await get_async_backend().run_sync_in_worker_thread(
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\anyio\_backends\_asyncio.py", line 2485, in run_sync_in_worker_thread
    return await future
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\anyio\_backends\_asyncio.py", line 976, in run
    result = context.run(func, *args)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\gradio\utils.py", line 915, in wrapper
    response = f(*args, **kwargs)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\run_gradio.py", line 239, in on_submit_single
    grey_depth_filename, grey_depth_image = process_image(original_image, bit_depth, input_size)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\run_gradio.py", line 109, in process_image
    depth = predict_depth(image[:, :, ::-1], input_size)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\run_gradio.py", line 81, in predict_depth
    return model.infer_image(image, input_size)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\torch\utils\_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\depth_anything_v2\dpt.py", line 190, in infer_image
    depth = self.forward(image)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\depth_anything_v2\dpt.py", line 179, in forward
    features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\depth_anything_v2\dinov2.py", line 308, in get_intermediate_layers
    outputs = self._get_intermediate_layers_not_chunked(x, n)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\depth_anything_v2\dinov2.py", line 277, in _get_intermediate_layers_not_chunked
    x = blk(x)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\torch\nn\modules\module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\torch\nn\modules\module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\depth_anything_v2\dinov2_layers\block.py", line 247, in forward
    return super().forward(x_or_x_list)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\depth_anything_v2\dinov2_layers\block.py", line 105, in forward
    x = x + attn_residual_func(x)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\depth_anything_v2\dinov2_layers\block.py", line 84, in attn_residual_func
    return self.ls1(self.attn(self.norm1(x)))
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\torch\nn\modules\module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\venv\lib\site-packages\torch\nn\modules\module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\depth_anything_v2\dinov2_layers\attention.py", line 69, in forward
    return super().forward(x)
  File "F:\DEVstuFF\NEW\Upgraded-Depth-Anything-V2\depth_anything_v2\dinov2_layers\attention.py", line 54, in forward
    attn = q @ k.transpose(-2, -1)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 115.93 GiB. GPU
