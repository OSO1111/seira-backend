import os
import sys
import uuid
import time
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from convert import convert_model

# ============================================================
# App
# ============================================================
app = FastAPI(title="SEIRA Backend", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Output directory
# ============================================================
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "generated"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = OUTPUT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = OUTPUT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPAIRED_DIR = OUTPUT_DIR / "repaired"
REPAIRED_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

# ============================================================
# In-memory Job Store (MVP용)
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
        "type": job_type,  # generate_images | generate_3d | repair_mesh
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


def _public_file_url(relative_path: str) -> str:
    relative_path = relative_path.lstrip("/").replace("\\", "/")
    public_base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    if public_base:
        return f"{public_base}/files/{relative_path}"
    return f"/files/{relative_path}"


def _safe_filename(filename: str) -> str:
    name = Path(filename or "uploaded_model.glb").name
    name = name.replace(" ", "_")
    return name


def _python_bin() -> str:
    return sys.executable


def _run_python_json(script_path: str, *args: str) -> Dict[str, Any]:
    result = subprocess.run(
        [_python_bin(), script_path, *args],
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = (result.stdout or "").strip()
    if not stdout:
        raise RuntimeError(f"{script_path} returned empty stdout")

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{script_path} did not return valid JSON. stdout={stdout!r}, stderr={(result.stderr or '').strip()!r}"
        ) from e


def _run_python_nojson(script_path: str, *args: str) -> None:
    result = subprocess.run(
        [_python_bin(), script_path, *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{script_path} failed. stdout={(result.stdout or '').strip()!r}, stderr={(result.stderr or '').strip()!r}"
        )


# ============================================================
# Models
# ============================================================
class GenerateImagesRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    n: int = Field(4, ge=1, le=8)
    size: str = Field("1024x1024")


class GenerateImagesResponse(BaseModel):
    job_id: str
    status: str


class Generate3DRequest(BaseModel):
    prompt: Optional[str] = None
    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    output_format: str = Field("glb")  # glb / stl / obj / fbx
    label: Optional[str] = None


class Generate3DResponse(BaseModel):
    job_id: str
    status: str


class RepairMeshResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    type: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


# ============================================================
# OpenAI Image Generation
# ============================================================
def _openai_generate_images(prompt: str, n: int, size: str) -> List[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

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
    urls: List[str] = []

    for item in data.get("data", []):
        if "url" in item and item["url"]:
            urls.append(item["url"])
        elif "b64_json" in item and item["b64_json"]:
            b64 = item["b64_json"]
            urls.append(f"data:image/png;base64,{b64}")

    if not urls:
        raise RuntimeError("OpenAI returned no images")

    return urls


# ============================================================
# Tripo3D helpers
# ============================================================
TRIPO_BASE_URL = os.environ.get("TRIPO_BASE_URL", "https://api.tripo3d.ai/v2/openapi")
TRIPO_TIMEOUT = float(os.environ.get("TRIPO_TIMEOUT", "60"))
TRIPO_POLL_INTERVAL = float(os.environ.get("TRIPO_POLL_INTERVAL", "5"))
TRIPO_MAX_POLLS = int(os.environ.get("TRIPO_MAX_POLLS", "60"))


def _get_tripo_headers() -> Dict[str, str]:
    api_key = os.environ.get("TRIPO_API_KEY")
    if not api_key:
        raise RuntimeError("TRIPO_API_KEY is not set")

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _download_file(url: str, dest_path: Path) -> None:
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)


def _create_tripo_text_task(prompt: str) -> str:
    headers = _get_tripo_headers()
    url = f"{TRIPO_BASE_URL}/task"

    payload = {
        "type": "text_to_model",
        "prompt": prompt,
    }

    print("[TRIPO] CREATE URL:", url)
    print("[TRIPO] CREATE BODY:", payload)

    with httpx.Client(timeout=TRIPO_TIMEOUT) as client:
        response = client.post(url, headers=headers, json=payload)

    print("[TRIPO] CREATE STATUS:", response.status_code)
    print("[TRIPO] CREATE RESPONSE:", response.text)

    response.raise_for_status()
    data = response.json()

    task_id = data.get("data", {}).get("task_id") or data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Tripo create task: task_id not found in response: {data}")

    return task_id


def _create_tripo_image_task(
    image_url: Optional[str],
    image_b64: Optional[str],
    label: Optional[str],
) -> str:
    headers = _get_tripo_headers()
    url = f"{TRIPO_BASE_URL}/task"

    payload: Dict[str, Any] = {
        "type": "image_to_model",
    }

    if image_url:
        payload["file"] = {"type": "url", "url": image_url}
    elif image_b64:
        payload["file"] = {"type": "base64", "data": image_b64}
    else:
        raise RuntimeError("image_url or image_b64 is required")

    if label:
        payload["name"] = label

    print("[TRIPO] CREATE URL:", url)
    print("[TRIPO] CREATE BODY:", payload)

    with httpx.Client(timeout=TRIPO_TIMEOUT) as client:
        response = client.post(url, headers=headers, json=payload)

    print("[TRIPO] CREATE STATUS:", response.status_code)
    print("[TRIPO] CREATE RESPONSE:", response.text)

    response.raise_for_status()
    data = response.json()

    task_id = data.get("data", {}).get("task_id") or data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Tripo create task: task_id not found in response: {data}")

    return task_id


def _get_tripo_task(task_id: str) -> Dict[str, Any]:
    headers = _get_tripo_headers()
    url = f"{TRIPO_BASE_URL}/task/{task_id}"

    with httpx.Client(timeout=TRIPO_TIMEOUT) as client:
        response = client.get(url, headers=headers)

    print("[TRIPO] POLL URL:", url)
    print("[TRIPO] POLL STATUS:", response.status_code)
    print("[TRIPO] POLL RESPONSE:", response.text)

    response.raise_for_status()
    return response.json()


def _extract_model_url(task_data: Dict[str, Any]) -> Optional[str]:
    root = task_data.get("data", task_data)

    candidates = [
        root.get("output", {}).get("model_url"),
        root.get("output", {}).get("glb_url"),
        root.get("model_url"),
        root.get("glb_url"),
        root.get("result", {}).get("model_url"),
        root.get("result", {}).get("glb_url"),
    ]

    for c in candidates:
        if c:
            return c

    return None


def _wait_for_tripo_result(task_id: str) -> str:
    for _ in range(TRIPO_MAX_POLLS):
        task_data = _get_tripo_task(task_id)
        root = task_data.get("data", task_data)

        status = (
            root.get("status")
            or root.get("task_status")
            or root.get("state")
            or ""
        ).lower()

        if status in {"success", "completed", "done"}:
            model_url = _extract_model_url(task_data)
            if not model_url:
                raise RuntimeError(f"Tripo task completed but no model URL found: {task_data}")
            return model_url

        if status in {"failed", "error"}:
            raise RuntimeError(f"Tripo task failed: {task_data}")

        time.sleep(TRIPO_POLL_INTERVAL)

    raise RuntimeError("Tripo task polling timed out")


# ============================================================
# Gemini helpers (optional for mesh repair planning)
# ============================================================
def _gemini_repair_plan(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    OPENAI_API_KEY가 없어도 돌아가게 기본 fallback 포함.
    나중에 Gemini/OpenAI whichever 쓰고 싶으면 이 함수만 갈아끼우면 됨.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_TEXT_MODEL", "gpt-5-mini")

    fallback = {
        "status": "repair_needed",
        "issues": analysis.get("issues", []),
        "repair_plan": [
            "remove_duplicate_faces",
            "remove_degenerate_faces",
            "keep_largest_component",
        ],
        "confidence": 0.5,
        "reason": "fallback plan",
    }

    if not api_key:
        return fallback

    try:
        # OpenAI Responses API 예시
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        prompt = (
            "You are a 3D mesh repair planner.\n"
            "Return JSON only.\n"
            "Allowed repair_plan steps: "
            "[remove_duplicate_faces, remove_degenerate_faces, keep_largest_component, "
            "merge_vertices, close_holes, manifold_rebuild, voxel_remesh]\n\n"
            f"analysis={json.dumps(analysis, ensure_ascii=False)}"
        )

        payload = {
            "model": model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "repair_plan",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "issues": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "repair_plan": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["status", "issues", "repair_plan", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
        }

        with httpx.Client(timeout=60) as client:
            r = client.post(url, headers=headers, json=payload)

        if r.status_code >= 400:
            return fallback

        data = r.json()
        output_text = data.get("output_text", "").strip()
        if not output_text:
            return fallback

        parsed = json.loads(output_text)
        if "repair_plan" not in parsed:
            return fallback
        return parsed

    except Exception:
        return fallback


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
                "images": urls,
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
    job = JOBS.get(job_id)
    if not job:
        return

    temp_dir = TMP_DIR / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        _set_job(job_id, status=JOB_STATUS_PROCESSING)

        payload = job["payload"]
        prompt = payload.get("prompt")
        image_url = payload.get("image_url")
        image_b64 = payload.get("image_b64")
        output_format = (payload.get("output_format") or "glb").lower()
        label = payload.get("label")

        if not prompt and not image_url and not image_b64:
            raise RuntimeError("prompt or image_url or image_b64 is required")

        # 1) Tripo task 생성
        if prompt:
            tripo_task_id = _create_tripo_text_task(prompt)
        else:
            tripo_task_id = _create_tripo_image_task(image_url, image_b64, label)

        # 2) polling해서 최종 GLB URL 받기
        glb_url = _wait_for_tripo_result(tripo_task_id)

        # 3) GLB 다운로드
        input_glb_path = temp_dir / f"{job_id}.glb"
        _download_file(glb_url, input_glb_path)

        # 4) 원하는 형식으로 변환
        final_filename = f"{job_id}.{output_format}"
        final_path = OUTPUT_DIR / final_filename

        convert_model(
            input_path=str(input_glb_path),
            output_path=str(final_path),
            output_format=output_format,
        )

        # 5) 결과 저장
        _set_job(
            job_id,
            status=JOB_STATUS_COMPLETED,
            result={
                "model_url": _public_file_url(final_filename),
                "source_glb_url": glb_url,
                "format": output_format,
                "tripo_task_id": tripo_task_id,
            },
        )

    except Exception as e:
        _set_job(
            job_id,
            status=JOB_STATUS_FAILED,
            error={"message": str(e)},
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def worker_repair_mesh(job_id: str) -> None:
    job = JOBS.get(job_id)
    if not job:
        return

    temp_dir = TMP_DIR / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        _set_job(job_id, status=JOB_STATUS_PROCESSING)

        payload = job["payload"]
        input_path = payload["input_path"]
        original_filename = payload["original_filename"]

        repaired_filename = f"{job_id}_{Path(original_filename).stem}.glb"
        repaired_relative_path = f"repaired/{repaired_filename}"
        repaired_abs_path = REPAIRED_DIR / repaired_filename

        # 1) analyze
        analysis = _run_python_json("mesh/analyze.py", input_path)

        # 2) get repair plan
        repair_plan_obj = _gemini_repair_plan(analysis)
        repair_plan = repair_plan_obj.get("repair_plan", [])

        # 3) repair
        _run_python_nojson(
            "mesh/repair.py",
            input_path,
            str(repaired_abs_path),
            json.dumps(repair_plan, ensure_ascii=False),
        )

        # 4) validate
        validation = _run_python_json("mesh/validate.py", str(repaired_abs_path))

        # 5) result
        _set_job(
            job_id,
            status=JOB_STATUS_COMPLETED,
            result={
                "original_filename": original_filename,
                "uploaded_model_url": _public_file_url(f"uploads/{Path(input_path).name}"),
                "repaired_model_url": _public_file_url(repaired_relative_path),
                "analysis": analysis,
                "repair_plan": repair_plan_obj,
                "validation": validation,
            },
        )

    except Exception as e:
        _set_job(
            job_id,
            status=JOB_STATUS_FAILED,
            error={"message": str(e)},
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# Routes
# ============================================================
@app.get("/health")
def health():
    return {"ok": True}


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


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "type": job["type"],
        "result": job["result"],
        "error": job["error"],
    }


@app.post("/v1/generate-3d", response_model=Generate3DResponse)
def generate_3d(req: Generate3DRequest, bg: BackgroundTasks):
    job = _new_job(
        job_type="generate_3d",
        payload={
            "prompt": req.prompt,
            "image_url": req.image_url,
            "image_b64": req.image_b64,
            "output_format": req.output_format,
            "label": req.label,
        },
    )
    bg.add_task(worker_generate_3d, job["job_id"])
    return {"job_id": job["job_id"], "status": job["status"]}


@app.post("/v1/repair-mesh", response_model=RepairMeshResponse)
async def repair_mesh(bg: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")

    ext = Path(file.filename).suffix.lower()
    allowed_exts = {".glb", ".stl", ".obj", ".ply", ".off"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type: {ext}. allowed={sorted(allowed_exts)}",
        )

    safe_name = _safe_filename(file.filename)
    saved_name = f"{uuid.uuid4()}_{safe_name}"
    saved_path = UPLOAD_DIR / saved_name

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job = _new_job(
        job_type="repair_mesh",
        payload={
            "input_path": str(saved_path),
            "original_filename": safe_name,
        },
    )

    bg.add_task(worker_repair_mesh, job["job_id"])
    return {"job_id": job["job_id"], "status": job["status"]}
