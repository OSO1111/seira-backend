import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================
# App
# =========================
app = FastAPI(title="SEIRA Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 시 WeWeb 도메인으로 제한 권장
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
# (운영에서는 Redis / DB로 교체)
# =========================
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

async def job_create(job_id: str):
    async with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": None,
            "result": {
                "preview_url": None,
                "glb_url": None,
                "stl_url": None,
            },
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }

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

# =========================
# Background Pipeline
# =========================
async def pipeline_generate_3d(job_id: str, user_text: str, quality: str):
    """
    실제 Tripo / 3D 생성 / STL 변환은
    아래 TODO 구간에 연결하면 됩니다.
    """
    try:
        await job_set(job_id, status="running", progress=5, message="Job started")

        # ---- Step 1: 요청 준비 ----
        await asyncio.sleep(1.0)
        await job_set(job_id, progress=25, message="Preparing generation request")

        # ---- Step 2: 3D 생성 (Stub) ----
        await asyncio.sleep(2.0)
        await job_set(job_id, progress=55, message="Generating 3D model")

        # ---- Step 3: 후처리 & STL 생성 (Stub) ----
        await asyncio.sleep(1.5)
        await job_set(job_id, progress=80, message="Converting to STL")

        # =========================
        # TODO (실제 구현 위치)
        # - Tripo API 호출
        # - glb 수신
        # - convert.py로 STL 변환
        # - S3 / R2 / Supabase Storage 업로드
        # =========================

        # 임시 결과 (프론트 연동 검증용)
        result = {
            "preview_url": f"https://picsum.photos/seed/{job_id}/512/512",
            "glb_url": f"https://example.com/generated/{job_id}.glb",
            "stl_url": f"https://example.com/generated/{job_id}.stl",
        }

        await job_set(
            job_id,
            status="done",
            progress=100,
            message="Done",
            result=result,
        )

    except Exception as e:
        await job_set(
            job_id,
            status="failed",
            progress=0,
            message=f"Unhandled error: {e}",
        )

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/generate-3d", response_model=Generate3DResponse)
async def generate_3d(req: Generate3DRequest):
    job_id = str(uuid.uuid4())
    quality = req.options.quality if req.options else "standard"

    await job_create(job_id)
    asyncio.create_task(
        pipeline_generate_3d(
            job_id=job_id,
            user_text=req.user_text,
            quality=quality,
        )
    )

    return Generate3DResponse(job_id=job_id, status="queued")

@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = await job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    # dict → Pydantic 모델로 안전 변환
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress", 0),
        message=job.get("message"),
        result=JobResult(**(job.get("result") or {})),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )
