import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

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
# Utils / Store
# =========================
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()

FILES_DIR = os.environ.get("FILES_DIR", "/tmp/seira_files")

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dirs():
    os.makedirs(FILES_DIR, exist_ok=True)

def get_openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()

def get_tripo_api_key() -> str:
    return os.environ.get("TRIPO_API_KEY", "").strip()

async def job_create(job_id: str, initial: Dict[str, Any]):
    async with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            **initial,
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
# Models
# =========================
class GenerateImagesRequest(BaseModel):
    prompt: str
    n: int = Field(default=4, ge=1, le=8)
    size: str = Field(default="512x512")  # OpenAI Image size 문자열 (데모)

class GeneratedImageItem(BaseModel):
    image_id: str
    url: str

class GenerateImagesResponse(BaseModel):
    job_id: str
    status: str
    images: List[GeneratedImageItem]

class GenerateOptions(BaseModel):
    quality: str = Field(default="standard", description="draft|standard|high")

class Generate3DFromImageRequest(BaseModel):
    image_url: str
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
# Files serving
# =========================
@app.get("/files/{job_id}/{filename}")
def serve_file(job_id: str, filename: str):
    path = os.path.join(FILES_DIR, job_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)

def public_file_url(job_id: str, filename: str) -> str:
    # Railway 배포 도메인이 바뀔 수 있으니, 프록시 환경이 아니라면 BASE_URL을 변수로 주는 게 안전합니다.
    base = os.environ.get("PUBLIC_BASE_URL", "").strip()
    if not base:
        # fallback: 상대 URL (WeWeb에서 같은 도메인 호출이면 OK)
        return f"/files/{job_id}/{filename}"
    return f"{base}/files/{job_id}/{filename}"

# =========================
# OpenAI image generation (optional)
# - 키가 없으면 picsum으로 fallback
# =========================
async def generate_images(prompt: str, n: int, size: str) -> List[str]:
    """
    returns list of image URLs (served by this backend or external fallback)
    """
    ensure_dirs()

    api_key = get_openai_api_key()
    if not api_key:
        # fallback: external placeholder
        return [f"https://picsum.photos/seed/{uuid.uuid4().hex}/512/512" for _ in range(n)]

    # OpenAI Images API (HTTP call) - returns base64; we save to file and serve via /files
    # NOTE: If your OpenAI image endpoint differs, adjust here.
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "n": n,
        "size": size,
        # "response_format": "b64_json"  # 일부 환경에서 필요
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            # fallback으로라도 UX 테스트가 가능하게 처리
            return [f"https://picsum.photos/seed/{uuid.uuid4().hex}/512/512" for _ in range(n)]

        data = resp.json()
        # OpenAI Images 응답은 보통 data:[{b64_json:...}] 형태
        items = data.get("data", [])
        urls: List[str] = []

        # 저장 job_id는 호출자에서 폴더 만들기 위해 필요하므로 여기선 URL만 반환하고,
        # 실제 저장은 상위에서 job_id를 알아야 합니다.
        # -> 상위에서 b64를 다시 받는 구조보다 단순화를 위해 여기서 "external url"만 쓰는 편이 더 쉬우나,
        #    현재는 b64 저장을 위해 임시로 "raw dict"를 반환해야 합니다.
        # 그래서 여기서는 예외적으로 b64를 그대로 return하지 않고,
        # generate_images()는 fallback url만 책임지고,
        # 실제 저장은 /v1/generate-images에서 처리합니다.
        #
        # 즉, 여기 도달하면 /v1/generate-images에서 같은 API를 한 번 더 호출하지 않도록
        # generate_images()를 쓰지 않고 아래 route에서 직접 호출하겠습니다.
        return []

# =========================
# Background: 3D pipeline (데모)
# =========================
async def pipeline_generate_3d_from_image(job_id: str, image_url: str, quality: str):
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

        await job_set(job_id, progress=20, message="Downloading image")
        await asyncio.sleep(0.3)

        # 실제 구현 단계:
        # 1) image_url 다운로드 -> 파일
        # 2) Tripo 3D 생성 호출 (image-to-3d)
        # 3) 결과 glb 다운로드
        # 4) convert.py로 stl 변환
        # 5) 업로드(S3/R2) 또는 /files로 서빙

        # 데모/플로우 확인용 stub:
        await job_set(job_id, progress=55, message="Generating 3D (stub)")
        await asyncio.sleep(1.2)

        await job_set(job_id, progress=85, message="Converting to STL (stub)")
        await asyncio.sleep(0.8)

        # 지금은 URL만 형태 맞춰서 내려줌 (Step 3에서 실제 파일 URL로 교체)
        result = {
            "preview_url": image_url,  # 임시: 선택한 이미지로 preview
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

@app.post("/v1/generate-images", response_model=GenerateImagesResponse)
async def generate_images_route(req: GenerateImagesRequest):
    ensure_dirs()

    job_id = str(uuid.uuid4())
    await job_create(
        job_id,
        {
            "type": "images",
            "status": "done",
            "progress": 100,
            "message": "Images ready",
            "result": {"preview_url": None, "glb_url": None, "stl_url": None},
        },
    )

    api_key = get_openai_api_key()

    # 1) 키가 없으면 placeholder 4장
    if not api_key:
        images = []
        for i in range(req.n):
            image_id = str(uuid.uuid4())
            images.append(
                {"image_id": image_id, "url": f"https://picsum.photos/seed/{image_id}/512/512"}
            )
        await job_set(job_id, images=images)
        return {"job_id": job_id, "status": "done", "images": images}

    # 2) OpenAI Images API 호출 -> b64 받아서 /files로 서빙
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-image-1", "prompt": req.prompt, "n": req.n, "size": req.size}

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            # 실패해도 UX는 진행되게 placeholder로 대체
            images = []
            for i in range(req.n):
                image_id = str(uuid.uuid4())
                images.append(
                    {"image_id": image_id, "url": f"https://picsum.photos/seed/{image_id}/512/512"}
                )
            await job_set(job_id, images=images, message=f"OpenAI image failed -> fallback ({resp.status_code})")
            return {"job_id": job_id, "status": "done", "images": images}

        data = resp.json()
        items = data.get("data", [])

        # 저장 폴더
        job_dir = os.path.join(FILES_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        images: List[Dict[str, str]] = []
        for idx, item in enumerate(items):
            image_id = str(uuid.uuid4())

            b64 = item.get("b64_json")
            # 혹시 url이 오는 형태면 그대로 사용
            direct_url = item.get("url")

            if direct_url:
                images.append({"image_id": image_id, "url": direct_url})
                continue

            if not b64:
                # 예외 fallback
                images.append({"image_id": image_id, "url": f"https://picsum.photos/seed/{image_id}/512/512"})
                continue

            # b64 저장
            import base64
            png_bytes = base64.b64decode(b64)
            filename = f"img_{idx+1}.png"
            path = os.path.join(job_dir, filename)
            with open(path, "wb") as f:
                f.write(png_bytes)

            images.append({"image_id": image_id, "url": public_file_url(job_id, filename)})

        await job_set(job_id, images=images)
        return {"job_id": job_id, "status": "done", "images": images}

@app.post("/v1/generate-3d-from-image", response_model=Generate3DResponse)
async def generate_3d_from_image(req: Generate3DFromImageRequest):
    job_id = str(uuid.uuid4())
    quality = (req.options.quality if req.options else "standard")

    await job_create(
        job_id,
        {
            "type": "3d",
            "status": "queued",
            "progress": 0,
            "message": None,
            "result": {"preview_url": None, "glb_url": None, "stl_url": None},
        },
    )

    asyncio.create_task(
        pipeline_generate_3d_from_image(
            job_id=job_id,
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

    # Pydantic 변환 안전하게( result 누락 대비 )
    result = job.get("result") or {}
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job.get("status", "queued"),
        progress=job.get("progress", 0),
        message=job.get("message"),
        result=JobResult(**result),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )
