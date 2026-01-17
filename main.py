# main.py
import os
import uuid
import json
import time
import asyncio
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# 외부 SDK들은 "서버 부팅"은 항상 성공해야 하므로
# 여기서 import는 하되, 클라이언트 생성은 절대 전역에서 하지 않습니다.
from openai import OpenAI
from tripo3d import TripoClient


# =========================
# App & CORS
# =========================
app = FastAPI(title="SEIRA Backend", version="1.0.0")

# WeWeb에서 호출할 것이므로 CORS 허용 (일단 전체 허용, 운영에서는 도메인 제한 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: 운영에서는 WeWeb 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Config
# =========================
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")  # Railway public domain을 넣으면 URL 생성이 깔끔합니다.
DATA_DIR = os.getenv("DATA_DIR", "/tmp/seira")    # Railway에서도 /tmp는 사용 가능 (영구 저장 아님)
os.makedirs(DATA_DIR, exist_ok=True)

# (선택) 작업 상태 메모리 저장: 간단한 PoC용
# 운영에서는 Redis/DB 권장
JOBS: Dict[str, Dict[str, Any]] = {}


# =========================
# Helpers: API clients
# =========================
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)

def get_tripo_client() -> TripoClient:
    api_key = os.getenv("TRIPO_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="TRIPO_API_KEY is not set")
    return TripoClient(api_key=api_key)


# =========================
# Models
# =========================
class Generate3DRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for 3D generation")
    # 필요하면 옵션 확장
    # quality: Optional[str] = "standard"

class Generate3DResponse(BaseModel):
    job_id: str
    status: str
    preview_glb_url: Optional[str] = None
    stl_url: Optional[str] = None
    glb_url: Optional[str] = None
    message: Optional[str] = None


# =========================
# Health
# =========================
@app.get("/health")
def health():
    return {"ok": True}


# =========================
# Files serving
# =========================
@app.get("/files/{job_id}/{filename}")
def get_file(job_id: str, filename: str):
    # /tmp 기반 파일 제공 (PoC)
    job_dir = os.path.join(DATA_DIR, job_id)
    file_path = os.path.join(job_dir, filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # content-type 자동 추정은 FileResponse가 해줍니다.
    return FileResponse(file_path)


def make_public_url(path: str) -> str:
    """
    path: "/files/...." 형태
    BASE_URL이 있으면 절대 URL로 반환, 없으면 path만 반환
    """
    if BASE_URL:
        return f"{BASE_URL}{path}"
    return path


# =========================
# Core: Background job
# =========================
async def run_generate_3d_job(job_id: str, prompt: str):
    """
    1) Tripo로 3D 생성 (예: glb 획득)
    2) 필요하면 convert.py로 stl 변환
    3) 결과 파일을 /tmp/seira/{job_id}/ 아래에 저장
    """
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["updated_at"] = time.time()

    job_dir = os.path.join(DATA_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        tripo = get_tripo_client()

        # -----------------------
        # (1) Tripo 3D generation
        # -----------------------
        # tripo3d 라이브러리의 실제 메서드명/리턴은 버전에 따라 차이가 있을 수 있습니다.
        # 사용 중인 SDK 기준으로 아래 블록만 맞춰주시면 됩니다.
        #
        # 아래는 "형태"만 잡아둔 예시입니다.
        #
        # 결과적으로 glb_bytes 혹은 glb_url을 얻어서 파일로 저장하면 됩니다.

        # 예시: SDK가 task 기반일 경우
        # task = tripo.text_to_3d(prompt=prompt)
        # task_id = task.task_id
        # while True:
        #     status = tripo.get_task(task_id)
        #     if status.status in ("succeeded", "failed"):
        #         break
        #     await asyncio.sleep(2)
        # if status.status == "failed":
        #     raise RuntimeError(status.error or "Tripo generation failed")
        # glb_url = status.output.glb

        # -----
        # 여기서는 사용자가 이미 tripo3d를 설치했고,
        # "실제 실행"은 사용자의 SDK 메서드에 맞춰 수정해야 하므로,
        # 우선은 실패하지 않도록 안내 에러를 넣어둡니다.
        # -----
        # 사용 중인 tripo3d 호출 코드가 이미 있다면 여기만 교체하세요.
        raise RuntimeError(
            "Tripo generation call is not wired yet. "
            "Replace the Tripo SDK call block in run_generate_3d_job() with your working code."
        )

        # (가정) glb_url을 얻었다면:
        # glb_path = os.path.join(job_dir, "model.glb")
        # # 다운로드 로직 (aiohttp 등)
        # await download_to_file(glb_url, glb_path)

        # -----------------------
        # (2) Convert GLB -> STL (optional)
        # -----------------------
        # convert.py가 있는 경우에만, 요청 시점에 import하여 변환합니다.
        # (서버 부팅 단계에서 convert import 에러로 죽는 것 방지)
        #
        # stl_path = os.path.join(job_dir, "model.stl")
        # try:
        #     from convert import glb_to_stl_via_obj  # 또는 generate_final_stl 등
        #     glb_to_stl_via_obj(glb_path, stl_path)
        # except Exception as e:
        #     # 변환 실패해도 glb 프리뷰는 제공 가능하므로, stl만 실패 처리
        #     JOBS[job_id]["warnings"] = [f"STL convert failed: {str(e)}"]

        # -----------------------
        # (3) Mark done
        # -----------------------
        # JOBS[job_id]["status"] = "done"
        # JOBS[job_id]["glb_filename"] = "model.glb"
        # JOBS[job_id]["stl_filename"] = "model.stl" if os.path.exists(stl_path) else None

    except HTTPException as he:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = he.detail
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
    finally:
        JOBS[job_id]["updated_at"] = time.time()


# =========================
# API: create job (async)
# =========================
@app.post("/v1/generate-3d", response_model=Generate3DResponse)
async def generate_3d(req: Generate3DRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "prompt": req.prompt,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    # 백그라운드에서 실행
    background_tasks.add_task(run_generate_3d_job, job_id, req.prompt)

    return Generate3DResponse(
        job_id=job_id,
        status="queued",
        message="Job queued"
    )


# =========================
# API: job status
# =========================
@app.get("/v1/jobs/{job_id}", response_model=Generate3DResponse)
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job["status"]

    glb_url = None
    stl_url = None
    preview_glb_url = None

    if status == "done":
        # 파일 URL 구성
        if job.get("glb_filename"):
            glb_url = make_public_url(f"/files/{job_id}/{job['glb_filename']}")
            preview_glb_url = glb_url  # 프리뷰는 glb를 그대로 사용
        if job.get("stl_filename"):
            stl_url = make_public_url(f"/files/{job_id}/{job['stl_filename']}")

    return Generate3DResponse(
        job_id=job_id,
        status=status,
        preview_glb_url=preview_glb_url,
        glb_url=glb_url,
        stl_url=stl_url,
        message=job.get("error") or (job.get("warnings")[0] if job.get("warnings") else None)
    )


# =========================
# (Optional) Simple root
# =========================
@app.get("/")
def root():
    return {
        "service": "seira-backend",
        "health": "/health",
        "generate_3d": "/v1/generate-3d",
        "job_status": "/v1/jobs/{job_id}",
    }
