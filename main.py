import os
import uuid
import time
import base64
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# App
# ============================================================
app = FastAPI(title="SEIRA Backend", version="1.0.0")

# CORS (WeWeb/프론트에서 직접 호출 or 프록시 호출 모두 대비)
# 배포 단계에서는 allow_origins를 WeWeb 도메인으로 좁히는 걸 권장합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# In-memory Job Store (간단 MVP용)
#  - Railway 재시작/스케일링 시 날아갑니다.
#  - 운영 단계에서는 Redis/DB로 옮기시는 걸 권장합니다.
# ============================================================
JOBS: Dict[str, Dict[str, Any]] = {}

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_PROCESSING = "processing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"


def _now_ts() -> float:
    return time.time()


def _new_job(job_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "type": job_type,  # "generate_images" | "generate_3d"
        "status": JOB_STATUS_QUEUED,
        "created_at": _now_ts(),
        "updated_at": _now_ts(),
        "payload": payload,
        "result": None,
        "error": None,
    }
    JOBS[job_id] = job
    return job


def _set_job(job_id: str, **kwargs) -> None:
    job = JOBS.get(job_id)
    if not job:
        return
    job.update(kwargs)
    job["updated_at"] = _now_ts()


# ============================================================
# Models
# ============================================================
class GenerateImagesRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    n: int = Field(4, ge=1, le=8)
    size: str = Field("1024x1024")  # OpenAI Images 기준
    # 필요 시 style/seed 등 확장 가능


class GenerateImagesResponse(BaseModel):
    job_id: str
    status: str


class Generate3DRequest(BaseModel):
    # 사용자가 선택한 이미지(보통 URL) 1장을 넣는 형태를 권장
    image_url: Optional[str] = None
    image_b64: Optional[str] = None  # base64 raw (dataURL 말고 "AAAA..." 형태 권장)
    # 3D 파이프라인 옵션 (필요 시 확장)
    output_format: str = Field("glb")  # glb/stl 등
    label: Optional[str] = None


class Generate3DResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    type: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


# ============================================================
# OpenAI Image Generation (가장 단순한 형태)
#  - 아래는 "실제로 동작하는 구현"을 위해 HTTP 호출로 작성했습니다.
#  - OPENAI_API_KEY 반드시 Railway 환경변수로 넣으셔야 합니다.
# ============================================================
def _openai_generate_images(prompt: str, n: int, size: str) -> List[str]:
    """
    Returns: list of image URLs (문자열)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # OpenAI Images API (Responses API가 아니라 Images API로 단순 처리)
    # - 모델은 환경변수로 바꿀 수 있게 처리
    # - 참고: 모델/엔드포인트는 향후 변경될 수 있으니 운영 시 최신 문서 기준으로 업데이트 권장
    import httpx

    model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")

    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
    }

    with httpx.Client(timeout=120) as client:
        r = client.post(url, headers=headers, json=payload)

    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI image generation failed: {r.status_code} {r.text}")

    data = r.json()
    # 일반적으로 data["data"] 안에 url 또는 b64_json이 옵니다.
    urls: List[str] = []
    for item in data.get("data", []):
        if "url" in item and item["url"]:
            urls.append(item["url"])
        elif "b64_json" in item and item["b64_json"]:
            # WeWeb에 바로 보여주려면 data URL로 변환
            b64 = item["b64_json"]
            urls.append(f"data:image/png;base64,{b64}")

    if not urls:
        raise RuntimeError("OpenAI returned no images")

    return urls


# ============================================================
# Background Workers
# ============================================================
def worker_generate_images(job_id: str) -> None:
    job = JOBS.get(job_id)
    if not job:
        return

    try:
        _set_job(job_id, status=JOB_STATUS_PROCESSING)

        payload = job["payload"]
        prompt = payload["prompt"]
        n = payload["n"]
        size = payload["size"]

        urls = _openai_generate_images(prompt=prompt, n=n, size=size)

        _set_job(
            job_id,
            status=JOB_STATUS_COMPLETED,
            result={
                "images": urls,  # ✅ WeWeb에서 이 배열을 그대로 반복 렌더링하면 됩니다.
                "count": len(urls),
            },
        )
    except Exception as e:
        _set_job(
            job_id,
            status=JOB_STATUS_FAILED,
            error={"message": str(e)},
        )


def worker_generate_3d(job_id: str) -> None:
    """
    여기서는 3D 생성 파이프라인이 실제로 어디에 있느냐(Tripo/자체 Blender/등)에 따라 달라집니다.
    일단 "구조"만 잡고, 실제 생성은 나중에 붙이실 수 있도록 fail-safe 형태로 둡니다.
    """
    job = JOBS.get(job_id)
    if not job:
        return

    try:
        _set_job(job_id, status=JOB_STATUS_PROCESSING)

        payload = job["payload"]
        image_url = payload.get("image_url")
        image_b64 = payload.get("image_b64")
        output_format = payload.get("output_format", "glb")

        if not image_url and not image_b64:
            raise RuntimeError("image_url or image_b64 is required")

        # TODO: 실제 3D 생성 로직 연결
        # 예: 외부 API 호출 → 결과 파일을 R2 업로드 → public URL 반환
        # 여기서는 데모로 "가짜 결과"를 반환합니다.
        # 운영 시 이 부분만 교체하시면 됩니다.

        fake_url = f"https://example.com/output/{job_id}.{output_format}"

        _set_job(
            job_id,
            status=JOB_STATUS_COMPLETED,
            result={
                "model_url": fake_url,
                "format": output_format,
            },
        )

    except Exception as e:
        _set_job(job_id, status=JOB_STATUS_FAILED, error={"message": str(e)})


# ============================================================
# Routes
# ============================================================
@app.get("/health")
def health():
    return {"ok": True}


# 1) 텍스트 -> 이미지 4장 생성 (Job 생성)
@app.post("/v1/generate-images", response_model=GenerateImagesResponse)
def generate_images(req: GenerateImagesRequest, bg: BackgroundTasks):
    job = _new_job(
        job_type="generate_images",
        payload={
            "prompt": req.prompt,
            "n": req.n,
            "size": req.size,
        },
    )
    bg.add_task(worker_generate_images, job["job_id"])
    return {"job_id": job["job_id"], "status": job["status"]}


# 2) Job Status 조회 (✅ 여기서 반드시 status를 내려줌)
@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    # ✅ WeWeb에서 필요로 하는 핵심: status / result(images or model_url)
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "type": job["type"],
        "result": job["result"],
        "error": job["error"],
    }


# 3) 선택한 이미지 1장 -> 3D 생성 (Job 생성)
@app.post("/v1/generate-3d", response_model=Generate3DResponse)
def generate_3d(req: Generate3DRequest, bg: BackgroundTasks):
    job = _new_job(
        job_type="generate_3d",
        payload={
            "image_url": req.image_url,
            "image_b64": req.image_b64,
            "output_format": req.output_format,
            "label": req.label,
        },
    )
    bg.add_task(worker_generate_3d, job["job_id"])
    return {"job_id": job["job_id"], "status": job["status"]}
