import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from mllm import batch_chat_completion, vl_ocr, batch_vl_ocr, chat_completion
from mpdf import extract_pdf_text_by_magic_pdf
from result import Result

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=100)


class ExtractPdfTextReq(BaseModel):
    pdf_path: str
    output_dir: str
    word_count_min: int = 80
    max_page: int = 8
    parse_method: Optional[str] = 'auto'


class ChatCompletionReq(BaseModel):
    model_name: str
    prompt: str


class BatchChatCompletionReq(BaseModel):
    model_name: str
    prompts: list[str]


class VlOcrReq(BaseModel):
    model_name: str
    prompt: str
    image_url: str


class BatchVlOcrReq(BaseModel):
    model_name: str
    prompt: str
    image_urls: list[str]


@app.post('/extract-pdf-text-accuracy')
async def extract_pdf_text_accuracy_api(req: ExtractPdfTextReq):
    text_path = await asyncio.get_event_loop().run_in_executor(
        executor,
        extract_pdf_text_by_magic_pdf,
        req.pdf_path,
        req.output_dir,
        req.word_count_min,
        req.max_page,
        req.parse_method
    )
    return JSONResponse(Result.success(text_path).to_dict(), status_code=200)


@app.post('/chat-completion')
async def chat_completion_api(req: ChatCompletionReq):
    response = await asyncio.get_event_loop().run_in_executor(
        executor,
        chat_completion,
        req.model_name,
        req.prompt
    )
    return JSONResponse(Result.success(response).to_dict(), status_code=200)


@app.post('/batch-chat-completion')
async def batch_chat_completion_api(req: BatchChatCompletionReq):
    response = await asyncio.get_event_loop().run_in_executor(
        executor,
        batch_chat_completion,
        req.model_name,
        req.prompts
    )
    return JSONResponse(Result.success(response).to_dict(), status_code=200)


@app.post('/vl-ocr')
async def vl_ocr_api(req: VlOcrReq):
    response = await asyncio.get_event_loop().run_in_executor(
        executor,
        vl_ocr,
        req.model_name,
        req.prompt,
        req.image_url
    )
    return JSONResponse(Result.success(response).to_dict(), status_code=200)


@app.post('/batch-vl-ocr')
async def batch_vl_ocr_api(req: BatchVlOcrReq):
    response = await asyncio.get_event_loop().run_in_executor(
        executor,
        batch_vl_ocr,
        req.model_name,
        req.prompt,
        req.image_urls
    )
    return JSONResponse(Result.success(response).to_dict(), status_code=200)


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, e: Exception):
    logger.error(f"Unhandled exception occurred: {e}")
    return JSONResponse(Result.fail(str(e)).to_dict(), status_code=200)


if __name__ == '__main__':
    port = os.getenv('FPS_GPU_PORT', '10000')
    uvicorn.run(app, host='0.0.0.0', port=int(port))
