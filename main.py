import os
import uuid
import time
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# =========================
# App
# =========================
app = FastAPI(title="SEIRA Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 운영에서는 WeWeb 도메인만 넣는 것을 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models (API Contract)
# =========================
class GenerateOptions(BaseModel):
    quality: str = Field(default="standard", description="draft|standard|high")

class Generate3DRequest(BaseModel):
    user_text: str
    options: Optional[GenerateOptions] = None

class Generate3DResponse(BaseModel):
    job_id: str
    status: str

class JobResult(BaseModel):
    preview_url: Optional[str] = None
    glb_url: Optional[str] = None
    stl_url: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    message: Optional[str] = None
    result: JobResult = Field(default_factory=JobResult)
    created_at: str
    updated_at: str

# =========================
# In-memory Job Store
# =========================
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

async def job_set(job_id: str, **kwargs):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(kwargs)
        job["updated_at"] = now_iso()

async def job_get(job_id: str) -> Optional[Dict[str, Any]]:
    async with JOBS_LOCK:
        return JOBS.get(job_id)

async def job_create(job_id: str):
    async with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": None,
            "result": {"preview_url": None, "glb_url": None, "stl_url": None},
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }

# =========================
# Config / Paths
# =========================
FILES_ROOT = os.environ.get("FILES_ROOT", "/tmp/seira_jobs")  # Railway에서 동작 확인용
os.makedirs(FILES_ROOT, exist_ok=True)

TRIPO_BASE_URL = os.environ.get("TRIPO_BASE_URL", "").strip()  # 예: https://api.tripo3d.ai (프로젝트에 맞게)
TRIPO_TIMEOUT = float(os.environ.get("TRIPO_TIMEOUT", "120"))
TRIPO_POLL_TIMEOUT_SEC = int(os.environ.get("TRIPO_POLL_TIMEOUT_SEC", "900"))
TRIPO_POLL_INTERVAL_SEC = float(os.environ.get("TRIPO_POLL_INTERVAL_SEC", "2.0"))

# =========================
# Lazy clients (중요)
# =========================
def get_openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()

def get_tripo_api_key() -> str:
    return os.environ.get("TRIPO_API_KEY", "").strip()

def job_dir(job_id: str) -> str:
    d = os.path.join(FILES_ROOT, job_id)
    os.makedirs(d, exist_ok=True)
    return d

def file_url(job_id: str, filename: str) -> str:
    # 같은 도메인에서 호출되면 상대경로로도 정상 동작합니다.
    return f"/v1/files/{job_id}/{filename}"

# =========================
# Convert (GLB -> STL)
# =========================
def glb_to_stl(glb_path: str, stl_path: str) -> None:
    """
    convert.py에 들어있는 GLB->STL 변환 함수를 호출합니다.
    함수명이 다르면 여기만 맞추시면 됩니다.
    """
    from convert import glb_to_stl_via_obj  # noqa: F401
    glb_to_stl_via_obj(glb_path, stl_path)

# =========================
# Tripo API helpers
# =========================
async def http_download(url: str, out_path: str, headers: Optional[dict] = None) -> None:
    headers = headers or {}
    async with httpx.AsyncClient(timeout=TRIPO_TIMEOUT, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)

async def tripo_create_job_http(prompt: str, quality: str) -> str:
    """
    Tripo 생성 요청. (HTTP 방식)
    실제 엔드포인트/필드가 다르면 여기만 수정하면 전체가 동작합니다.
    """
    api_key = get_tripo_api_key()
    if not api_key:
        raise ValueError("TRIPO_API_KEY is not set")

    if not TRIPO_BASE_URL:
        raise ValueError("TRIPO_BASE_URL is not set (set Railway Variables)")

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"prompt": prompt, "quality": quality}

    async with httpx.AsyncClient(timeout=TRIPO_TIMEOUT) as client:
        # TODO: Tripo 실제 엔드포인트에 맞게 수정
        r = await client.post(f"{TRIPO_BASE_URL}/v1/text-to-3d", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()

    # TODO: 실제 job id 필드에 맞게 수정
    # 예: data["job_id"] 또는 data["task_id"]
    if "job_id" in data:
        return data["job_id"]
    if "task_id" in data:
        return data["task_id"]
    raise ValueError(f"Unexpected Tripo response (no job id): {data}")

async def tripo_get_job_http(tripo_job_id: str) -> dict:
    """
    Tripo 상태 조회. (HTTP 방식)
    """
    api_key = get_tripo_api_key()
    if not api_key:
        raise ValueError("TRIPO_API_KEY is not set")

    if not TRIPO_BASE_URL:
        raise ValueError("TRIPO_BASE_URL is not set (set Railway Variables)")

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=TRIPO_TIMEOUT) as client:
        # TODO: Tripo 실제 엔드포인트에 맞게 수정
        r = await client.get(f"{TRIPO_BASE_URL}/v1/jobs/{tripo_job_id}", headers=headers)
        r.raise_for_status()
        data = r.json()

    return data

def extract_tripo_status(info: dict) -> str:
    # TODO: 실제 status 필드가 다르면 여기 수정
    return str(info.get("status", "")).lower()

def extract_tripo_result(info: dict) -> dict:
    """
    Tripo 응답에서 preview/glb/stl URL을 뽑아옵니다.
    필드명이 다르면 여기만 맞추면 됩니다.
    """
    # 다양한 형태를 커버
    result = info.get("result") or info.get("data") or info

    preview_url = (
        result.get("preview_url")
        or result.get("thumbnail_url")
        or result.get("image_url")
        or result.get("preview")
    )
    glb_url = (
        result.get("glb_url")
        or result.get("model_glb_url")
        or result.get("model_url")
        or result.get("glb")
    )
    stl_url = (
        result.get("stl_url")
        or result.get("model_stl_url")
        or result.get("stl")
    )
    return {"preview_url": preview_url, "glb_url": glb_url, "stl_url": stl_url}

# =========================
# Background pipeline (실제 GLB+STL 생성)
# =========================
async def pipeline_generate_3d(job_id: str, user_text: str, quality: str):
    try:
        await job_set(job_id, status="running", progress=5, message="Job started")

        # 키 체크
        tripo_key = get_tripo_api_key()
        if not tripo_key:
            await job_set(
                job_id,
                status="failed",
                progress=0,
                message="TRIPO_API_KEY is not set. Please set it in Railway Variables.",
            )
            return

        # OpenAI 키는 현재 파이프라인에 필수 아님(원하시면 체크 제거 가능)
        # openai_key = get_openai_api_key()
        # if not openai_key:
        #     await job_set(job_id, status="failed", progress=0,
        #                   message="OPENAI_API_KEY is not set. Please set it in Railway Variables.")
        #     return

        # 1) Tripo 생성 요청
        await job_set(job_id, progress=12, message="Submitting to Tripo")
        tripo_job_id = await tripo_create_job_http(prompt=user_text, quality=quality)

        # 2) Tripo 폴링
        await job_set(job_id, progress=20, message="Waiting Tripo job")
        t0 = time.time()

        remote_preview_url = None
        remote_glb_url = None
        remote_stl_url = None

        while True:
            if time.time() - t0 > TRIPO_POLL_TIMEOUT_SEC:
                await job_set(job_id, status="failed", progress=0, message="Tripo job timeout")
                return

            info = await tripo_get_job_http(tripo_job_id)
            status = extract_tripo_status(info)

            # 진행률 반영(있으면)
            p = info.get("progress")
            if isinstance(p, (int, float)):
                await job_set(job_id, progress=max(20, min(90, int(p))), message=f"Tripo: {status}")
            else:
                await job_set(job_id, message=f"Tripo: {status}")

            if status in ("succeeded", "success", "done", "completed"):
                r = extract_tripo_result(info)
                remote_preview_url = r.get("preview_url")
                remote_glb_url = r.get("glb_url")
                remote_stl_url = r.get("stl_url")
                break

            if status in ("failed", "error"):
                await job_set(job_id, status="failed", progress=0, message=f"Tripo failed: {info}")
                return

            await asyncio.sleep(TRIPO_POLL_INTERVAL_SEC)

        if not remote_glb_url:
            await job_set(job_id, status="failed", progress=0, message="Tripo result missing glb_url")
            return

        # 3) GLB 다운로드
        await job_set(job_id, progress=92, message="Downloading GLB")
        d = job_dir(job_id)
        glb_path = os.path.join(d, "model.glb")
        await http_download(remote_glb_url, glb_path)

        # 4) (선택) preview 이미지 다운로드해서 우리 서버에서 서빙
        preview_path = None
        if remote_preview_url:
            await job_set(job_id, progress=94, message="Downloading preview image")
            preview_path = os.path.join(d, "preview.png")
            try:
                await http_download(remote_preview_url, preview_path)
            except Exception:
                # 프리뷰 다운로드 실패해도 GLB/STL이 있으면 계속
                preview_path = None

        # 5) STL 변환 (convert.py 사용)
        await job_set(job_id, progress=96, message="Converting GLB -> STL")
        stl_path = os.path.join(d, "model.stl")
        await asyncio.to_thread(glb_to_stl, glb_path, stl_path)

        # 6) 결과 URL 세팅 (우리 서버 서빙 URL)
        result = {
            "preview_url": file_url(job_id, "preview.png") if preview_path else (remote_preview_url or None),
            "glb_url": file_url(job_id, "model.glb"),
            "stl_url": file_url(job_id, "model.stl"),
        }

        await job_set(job_id, status="done", progress=100, message="Done", result=result)

    except Exception as e:
        await job_set(job_id, status="failed", progress=0, message=f"Unhandled error: {e}")

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/generate-3d", response_model=Generate3DResponse)
async def generate_3d(req: Generate3DRequest):
    job_id = str(uuid.uuid4())
    quality = (req.options.quality if req.options else "standard")

    await job_create(job_id)
    asyncio.create_task(pipeline_generate_3d(job_id=job_id, user_text=req.user_text, quality=quality))

    return {"job_id": job_id, "status": "queued"}

@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = await job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    # dict -> Pydantic 안전 변환 (result 포함)
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress", 0),
        message=job.get("message"),
        result=JobResult(**(job.get("result") or {})),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )

@app.get("/v1/files/{job_id}/{filename}")
async def get_file(job_id: str, filename: str):
    # 디렉토리 트래버설 방지
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="invalid filename")

    path = os.path.join(job_dir(job_id), filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="file not found")

    # content-type은 FastAPI가 확장자로 추정합니다.
    return FileResponse(path)
