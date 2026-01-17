"""
main.py (Production-ready: R2 uploads + presigned download + 3D preview via GLB URL)

Endpoints
- GET  /health
- POST /parse
- POST /generate-image
- POST /generate-3d
- GET  /task/{task_id}
- GET  /download/{task_id}/{file_key}     -> returns presigned URL JSON

Required env vars
- OPENAI_API_KEY
- TRIPO_API_KEY
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_BUCKET
- R2_ENDPOINT   (ex: https://<account_id>.r2.cloudflarestorage.com)

Optional env vars
- WEWEB_ORIGINS (comma-separated)  ex: https://xxx.weweb.app,https://xxx.com
- PRESIGNED_EXPIRES (seconds) default 3600
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

from tripo3d import TripoClient
from convert import generate_final_stl


# -----------------------------
# App / Clients
# -----------------------------
app = FastAPI(title="SEIRA API", version="1.0.0")

openai_client = OpenAI()
tripo_client = TripoClient(api_key=os.environ.get("TRIPO_API_KEY", ""))

# R2 (S3-compatible)
def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

R2_ENDPOINT = _require_env("R2_ENDPOINT")
R2_BUCKET = _require_env("R2_BUCKET")
R2_ACCESS_KEY_ID = _require_env("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = _require_env("R2_SECRET_ACCESS_KEY")

r2 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)

PRESIGNED_EXPIRES = int(os.environ.get("PRESIGNED_EXPIRES", "3600"))

# CORS (WeWeb)
weweb_origins_raw = os.environ.get("WEWEB_ORIGINS", "").strip()
if weweb_origins_raw:
    allow_origins = [o.strip() for o in weweb_origins_raw.split(",") if o.strip()]
else:
    # 개발/테스트용. 운영에서는 WEWEB_ORIGINS를 반드시 지정하는 것을 권장합니다.
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class ParseRequest(BaseModel):
    user_text: str = Field(..., description="User prompt or description")

class ParseResponse(BaseModel):
    normalized_text: str
    extracted: Dict[str, Any] = Field(default_factory=dict)

class GenerateImageRequest(BaseModel):
    prompt: str
    size: str = Field(default="1024x1024", description="OpenAI image size")
    task_id: Optional[str] = None

class GenerateImageResponse(BaseModel):
    task_id: str
    image_key: str
    image_url: str
    meta_url: str

class Generate3DRequest(BaseModel):
    prompt: str
    # 필요 시 확장: style, seed, etc.
    task_id: Optional[str] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    created_at: str
    updated_at: str
    prompt: str
    files: Dict[str, str] = Field(default_factory=dict)   # file_key -> r2_object_key
    preview: Dict[str, str] = Field(default_factory=dict) # convenience: type->file_key
    error: Optional[str] = None


# -----------------------------
# Helpers (R2)
# -----------------------------
SAFE_FILEKEY_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def r2_put_json(key: str, data: Dict[str, Any]) -> None:
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    r2.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )

def r2_get_json(key: str) -> Dict[str, Any]:
    try:
        obj = r2.get_object(Bucket=R2_BUCKET, Key=key)
        raw = obj["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except r2.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Task not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read task meta: {e}")

def r2_upload_file(local_path: Path, key: str, content_type: Optional[str] = None) -> None:
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    r2.upload_file(str(local_path), R2_BUCKET, key, ExtraArgs=extra if extra else None)

def presigned_get_url(key: str, expires: int = PRESIGNED_EXPIRES) -> str:
    return r2.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET, "Key": key},
        ExpiresIn=expires,
    )

def task_meta_key(task_id: str) -> str:
    return f"tasks/{task_id}/meta.json"

def task_object_key(task_id: str, file_key: str) -> str:
    # stored in bucket
    return f"tasks/{task_id}/{file_key}"

def validate_file_key(file_key: str) -> None:
    if not SAFE_FILEKEY_RE.match(file_key) or ".." in file_key or file_key.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid file_key")

def init_task_meta(task_id: str, prompt: str) -> Dict[str, Any]:
    meta = {
        "task_id": task_id,
        "status": "queued",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "prompt": prompt,
        "files": {},    # file_key -> r2_object_key
        "preview": {},  # type -> file_key (ex: {"glb":"model.glb", "stl":"final.stl"})
        "error": None,
    }
    r2_put_json(task_meta_key(task_id), meta)
    return meta

def update_task_meta(task_id: str, **patch: Any) -> Dict[str, Any]:
    meta = r2_get_json(task_meta_key(task_id))
    meta.update(patch)
    meta["updated_at"] = now_iso()
    r2_put_json(task_meta_key(task_id), meta)
    return meta


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/parse", response_model=ParseResponse)
def parse(req: ParseRequest):
    """
    간단한 정규화/추출 샘플.
    필요하면 여기에서 OpenAI를 이용해 merch_type, label_text 등 구조화 추출로 확장 가능합니다.
    """
    text = req.user_text.strip()
    extracted: Dict[str, Any] = {}

    # 예시: 라벨 텍스트가 따옴표로 들어오면 추출
    # e.g., label: "JIMIN"
    m = re.search(r'["“”](.+?)["“”]', text)
    if m:
        extracted["quoted"] = m.group(1)

    return ParseResponse(normalized_text=text, extracted=extracted)


@app.post("/generate-image", response_model=GenerateImageResponse)
def generate_image(req: GenerateImageRequest):
    """
    OpenAI 이미지 생성 -> PNG 바이트를 R2에 업로드 -> presigned URL 반환
    """
    task_id = req.task_id or str(uuid.uuid4())
    meta = init_task_meta(task_id, prompt=req.prompt)

    update_task_meta(task_id, status="running")

    try:
        # OpenAI Images API (SDK 버전에 따라 응답 포맷이 다를 수 있습니다)
        # 본인 프로젝트에서 이미 이미지 생성이 동작하는 방식이 있으면 그 방식으로 교체하세요.
        img = openai_client.images.generate(
            model="gpt-image-1",
            prompt=req.prompt,
            size=req.size,
        )

        # gpt-image-1: b64_json 형태로 올 때가 많음
        b64 = img.data[0].b64_json
        import base64
        png_bytes = base64.b64decode(b64)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "image.png"
            out.write_bytes(png_bytes)

            file_key = "image.png"
            obj_key = task_object_key(task_id, file_key)
            r2_upload_file(out, obj_key, content_type="image/png")

        meta["files"][file_key] = obj_key
        meta["preview"]["image"] = file_key
        meta["status"] = "succeeded"
        meta["updated_at"] = now_iso()
        r2_put_json(task_meta_key(task_id), meta)

        image_url = presigned_get_url(obj_key)
        meta_url = presigned_get_url(task_meta_key(task_id))

        return GenerateImageResponse(
            task_id=task_id,
            image_key=file_key,
            image_url=image_url,
            meta_url=meta_url,
        )

    except Exception as e:
        update_task_meta(task_id, status="failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")


@app.post("/generate-3d")
def generate_3d(req: Generate3DRequest):
    """
    3D 생성 파이프라인:
    1) Tripo로 3D 생성(보통 glb/obj 등)
    2) convert.generate_final_stl 로 최종 STL 생성
    3) (프리뷰) GLB 파일을 R2에 업로드하고 presigned URL을 반환 -> WeWeb에서 model-viewer로 미리보기
    4) task meta.json은 R2에 저장
    """
    task_id = req.task_id or str(uuid.uuid4())
    init_task_meta(task_id, prompt=req.prompt)
    update_task_meta(task_id, status="running")

    try:
        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)

            # -------------------------
            # 1) Tripo 3D 생성
            # -------------------------
            # 주의: TripoClient SDK 버전에 따라 메서드/리턴이 다를 수 있습니다.
            # 현재 사용자 코드에서 tripo3d를 이미 사용하고 있다고 하셔서 기본 흐름만 잡아둡니다.
            #
            # 아래는 "개념적으로" glb를 얻는 위치입니다.
            # 실제 SDK에서 model_id 얻고 -> 결과 조회 -> glb 다운로드 식으로 맞추셔야 합니다.
            #
            # 예시(가정):
            # job = tripo_client.text_to_3d(prompt=req.prompt)
            # model_url = job["model_url"]  # 또는 job.model_url
            #
            # 여기서는 가능한 한 "download_url이 있다"는 가정으로 작성합니다.

            job = tripo_client.text_to_3d(req.prompt)  # <-- SDK에 맞게 필요 시 수정
            # job에서 다운로드 URL/ID를 얻는 부분을 프로젝트 상황에 맞게 조정하세요.
            model_glb_url = None
            model_id = None

            # 흔한 패턴들 커버(안전):
            if isinstance(job, dict):
                model_glb_url = job.get("glb_url") or job.get("model_glb_url") or job.get("url")
                model_id = job.get("id") or job.get("model_id")
            else:
                model_glb_url = getattr(job, "glb_url", None) or getattr(job, "model_glb_url", None) or getattr(job, "url", None)
                model_id = getattr(job, "id", None) or getattr(job, "model_id", None)

            # model_glb_url이 없으면, SDK의 "get 결과" 메서드로 폴링해야 할 수 있습니다.
            # 그 경우 아래 부분을 "result = tripo_client.get_model(model_id)" 같은 형태로 교체하세요.
            if not model_glb_url and model_id:
                # 예시(가정):
                # result = tripo_client.get_model(model_id)
                # model_glb_url = result.get("glb_url")
                result = tripo_client.get_model(model_id)  # <-- SDK에 맞게 수정 가능
                if isinstance(result, dict):
                    model_glb_url = result.get("glb_url") or result.get("model_glb_url") or result.get("url")

            if not model_glb_url:
                raise RuntimeError("Tripo result did not include a GLB download URL. Please adjust SDK mapping.")

            # -------------------------
            # 2) GLB 다운로드
            # -------------------------
            import aiohttp
            import asyncio

            async def _download(url: str, out_path: Path) -> None:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"Failed to download GLB: {resp.status}")
                        out_path.write_bytes(await resp.read())

            glb_path = workdir / "model.glb"
            asyncio.run(_download(model_glb_url, glb_path))

            # -------------------------
            # 3) STL 변환/후처리
            # -------------------------
            # generate_final_stl 은 사용자 프로젝트 convert.py의 함수에 맞춰 호출합니다.
            # 함수 시그니처가 다르면 이 부분만 맞춰 주세요.
            final_stl_path = workdir / "final.stl"
            generate_final_stl(str(glb_path), str(final_stl_path))  # <-- convert.py 시그니처에 맞게 조정 가능

            # -------------------------
            # 4) R2 업로드 (GLB + STL + meta)
            # -------------------------
            glb_file_key = "model.glb"
            stl_file_key = "final.stl"

            glb_obj_key = task_object_key(task_id, glb_file_key)
            stl_obj_key = task_object_key(task_id, stl_file_key)

            r2_upload_file(glb_path, glb_obj_key, content_type="model/gltf-binary")
            r2_upload_file(final_stl_path, stl_obj_key, content_type="model/stl")

        # meta 업데이트(임시 디렉토리 밖에서)
        meta = r2_get_json(task_meta_key(task_id))
        meta["files"][glb_file_key] = glb_obj_key
        meta["files"][stl_file_key] = stl_obj_key
        meta["preview"]["glb"] = glb_file_key
        meta["preview"]["stl"] = stl_file_key
        meta["status"] = "succeeded"
        meta["updated_at"] = now_iso()
        r2_put_json(task_meta_key(task_id), meta)

        # presigned URL 반환 (WeWeb 3D 미리보기용)
        glb_url = presigned_get_url(glb_obj_key)
        stl_url = presigned_get_url(stl_obj_key)
        meta_url = presigned_get_url(task_meta_key(task_id))

        return {
            "task_id": task_id,
            "status": "succeeded",
            "preview": {
                "glb_url": glb_url,     # WeWeb에서 3D viewer로 사용
                "stl_url": stl_url,     # 다운로드/후처리용
            },
            "files": {
                "model.glb": glb_file_key,
                "final.stl": stl_file_key,
            },
            "meta_url": meta_url,
            "expires_in": PRESIGNED_EXPIRES,
        }

    except Exception as e:
        update_task_meta(task_id, status="failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"3D generation failed: {e}")


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
def get_task(task_id: str):
    meta = r2_get_json(task_meta_key(task_id))
    return TaskStatusResponse(**meta)


@app.get("/download/{task_id}/{file_key}")
def download(task_id: str, file_key: str):
    """
    Returns a presigned URL for a file stored under tasks/{task_id}/{file_key}.
    """
    validate_file_key(file_key)
    obj_key = task_object_key(task_id, file_key)

    # 존재 확인(없으면 404)
    try:
        r2.head_object(Bucket=R2_BUCKET, Key=obj_key)
    except Exception:
        raise HTTPException(status_code=404, detail="File not found")

    url = presigned_get_url(obj_key)
    return {"url": url, "expires_in": PRESIGNED_EXPIRES, "key": obj_key}