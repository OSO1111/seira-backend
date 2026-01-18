import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================
# App
# =========================
app = FastAPI(title="SEIRA Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영에서는 WeWeb 도메인만 넣는 것을 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models (API Contract)
# =========================
class GenerateOptions(BaseModel):
    quality: str = Field(default="standard", description="draft|standard|high")


class GenerateImagesRequest(BaseModel):
    prompt: str
    n: int = Field(default=4, ge=1, le=8, description="number of images")


class GeneratedImage(BaseModel):
    id: str
    url: str


class GenerateImagesResponse(BaseModel):
    images: List[GeneratedImage]


class Generate3DRequest(BaseModel):
    # 기존 user_text 유지(호환성). 실제 사용은 image_url을 우선 권장.
    user_text: Optional[str] = None
    image_url: Optional[str] = None
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
# Config / Keys
# =========================
def get_openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()

def get_tripo_api_key() -> str:
    return os.environ.get("TRIPO_API_KEY", "").strip()

def get_public_base_url() -> str:
    """
    결과 URL을 절대경로로 만들 때 사용.
    Railway에서는 보통 https://<service>.up.railway.app 를 직접 넣는 방식 권장.
    """
    return os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")


# =========================
# Placeholder external calls
# (여기는 Step 3에서 실제 이미지 생성/3D 생성으로 교체)
# =========================
async def generate_images_stub(prompt: str, n: int) -> List[Dict[str, str]]:
    """
    TODO: 실제 이미지 생성 API로 교체.
    지금은 프론트 플로우 확인용으로 picsum을 반환.
    """
    images = []
    seed_base = uuid.uuid4().hex[:8]
    for i in range(n):
        img_id = f"{seed_base}-{i+1}"
        url = f"https://picsum.photos/seed/{img_id}/768/768"
        images.append({"id": img_id, "url": url})
    return images


async def pipeline_generate_3d(job_id: str, user_text: Optional[str], image_url: Optional[str], quality: str):
    """
    TODO(실전):
      1) (선택) 이미지 URL을 Tripo/Meshy 등에 전달 (Image-to-3D)
      2) 생성된 GLB 다운로드
      3) GLB -> STL 변환 (convert.py / Blender)
      4) R2/S3 업로드 후 공개 URL 획득
      5) result.preview_url / glb_url / stl_url 채우기
    """
    try:
        await job_set(job_id, status="running", progress=5, message="Job started")

        tripo_key = get_tripo_api_key()
        if not tripo_key:
            await job_set(
                job_id,
                status="failed",
                progress=0,
                message="TRIPO_API_KEY is not set. Please set it in Railway Variables.",
            )
            return

        if not image_url and not user_text:
            await job_set(
                job_id,
                status="failed",
                progress=0,
                message="image_url or user_text is required.",
            )
            return

        # 1) 진행(모의)
        await job_set(job_id, progress=20, message="Preparing 3D generation request")
        await asyncio.sleep(0.8)

        await job_set(job_id, progress=55, message="Generating 3D (stub)")
        await asyncio.sleep(1.2)

        await job_set(job_id, progress=80, message="Converting to STL (stub)")
        await asyncio.sleep(0.8)

        # 2) 결과 URL (임시) — Step 3에서 실제 결과로 교체
        # preview_url: 이미지 미리보기(선택한 이미지 또는 렌더 프리뷰)
        # glb_url/stl_url: 실제 파일 URL
        preview = image_url or f"https://picsum.photos/seed/{job_id}/768/768"

        # 다운로드 링크도 일단 "있어 보이게" 임시로 꽂아둠.
        # 실전에서는 R2/S3에 올린 URL로 바꾸세요.
        result = {
            "preview_url": preview,
            "glb_url": f"https://example.com/files/{job_id}.glb",
            "stl_url": f"https://example.com/files/{job_id}.stl",
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


@app.post("/v1/generate-images", response_model=GenerateImagesResponse)
async def generate_images(req: GenerateImagesRequest):
    """
    텍스트 -> 이미지 N장 생성
    WeWeb에서 리스트로 뿌리고, 사용자가 하나 선택하게 만드는 용도.
    """
    images = await generate_images_stub(prompt=req.prompt, n=req.n)
    return {"images": images}


@app.post("/v1/generate-3d", response_model=Generate3DResponse)
async def generate_3d(req: Generate3DRequest):
    """
    선택된 image_url(권장) 또는 user_text(호환) 기반으로 3D job 생성.
    """
    job_id = str(uuid.uuid4())
    quality = (req.options.quality if req.options else "standard")

    await job_create(job_id)

    # 백그라운드 실행
    asyncio.create_task(
        pipeline_generate_3d(
            job_id=job_id,
            user_text=req.user_text,
            image_url=req.image_url,
            quality=quality,
        )
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = await job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    # dict -> Pydantic model로 안전 변환
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress", 0),
        message=job.get("message"),
        result=JobResult(**(job.get("result") or {})),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )
