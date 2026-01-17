import os
import uuid
import time
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================
# App
# =========================
app = FastAPI(title="SEIRA Backend", version="0.1.0")

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
# (운영에서는 Redis/DB로 교체 권장)
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
# Lazy clients (중요)
# =========================
def get_openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()

def get_tripo_api_key() -> str:
    # 사용 중인 키 이름이 TRIPO_API_KEY로 보입니다.
    return os.environ.get("TRIPO_API_KEY", "").strip()

# =========================
# Background pipeline (Step 1/2는 "흐름" 완성이 목적)
# Step 3에서 실제 Tripo/convert/S3 업로드로 교체
# =========================
async def pipeline_generate_3d(job_id: str, user_text: str, quality: str):
    try:
        await job_set(job_id, status="running", progress=5, message="Job started")

        # 1) 키 확인 (없어도 서버는 살아있고, job만 failed 처리)
        tripo_key = get_tripo_api_key()
        openai_key = get_openai_api_key()

        if not tripo_key:
            await job_set(
                job_id,
                status="failed",
                progress=0,
                message="TRIPO_API_KEY is not set. Please set it in Railway Variables.",
            )
            return

        if not openai_key:
            # OpenAI를 꼭 쓰는 구조가 아니라면 경고만 하고 계속 진행해도 됩니다.
            # 지금은 명확히 실패 처리(원하시면 warning으로 변경 가능)
            await job_set(
                job_id,
                status="failed",
                progress=0,
                message="OPENAI_API_KEY is not set. Please set it in Railway Variables.",
            )
            return

        # 2) (현재 단계) 실제 생성은 아직 붙이기 전이라 "모의 진행"만 합니다.
        # Step 3에서 여기를 Tripo 호출 + convert + 업로드로 교체합니다.
        await job_set(job_id, progress=25, message="Preparing generation request")
        await asyncio.sleep(1.0)

        await job_set(job_id, progress=55, message="Generating 3D (stub)")
        await asyncio.sleep(1.5)

        await job_set(job_id, progress=80, message="Post-processing (stub)")
        await asyncio.sleep(1.0)

        # 3) 결과 URL (Step 3에서 실제 URL로 채움)
        # 지금은 done까지 가는 “프론트 플로우 확인”용
        # 3) 결과 URL (Step 3에서 실제 URL로 채움)
        # 지금은 done까지 가는 “프론트 플로우 확인”용 (임시 이미지)
        result = {
            "preview_url": f"https://picsum.photos/seed/{job_id}/512/512",
            "glb_url": None,
            "stl_url": None,
        }
        await job_set(job_id, status="done", progress=100, message="Done (stub)", result=result)

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

    # 백그라운드 실행
    asyncio.create_task(pipeline_generate_3d(job_id=job_id, user_text=req.user_text, quality=quality))

    return {"job_id": job_id, "status": "queued"}

@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = await job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    # dict -> Pydantic model (result 중첩 포함)로 안전 변환
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress", 0),
        message=job.get("message"),
        result=JobResult(**(job.get("result") or {})),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )
