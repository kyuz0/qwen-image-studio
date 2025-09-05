import asyncio
import json
import os
import sys
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import json as _json
import subprocess
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

running_processes: Dict[str, asyncio.subprocess.Process] = {}

# --- basic paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PUBLIC_DIR = HERE / "static"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

# uploads
UPLOAD_DIR = Path(tempfile.gettempdir()) / "qwen-image-studio"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- core manager (in-process)
SRC_DIR = (PROJECT_ROOT / "src")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen_image_mps.core import get_manager
manager = get_manager()

app = FastAPI()

@app.on_event("startup")
async def _warmup():
    # preloads generate+ultra by default (GPU, bf16)
    await asyncio.to_thread(manager.warmup, None)

# IMPORTANT: serve static at /static (NOT at "/")
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

# --- job store
# persistence
STATE_DIR = Path.home() / ".qwen-image-studio"
STATE_DIR.mkdir(parents=True, exist_ok=True)
(STATE_DIR / "jobs").mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "jobs.json"

def _atomic_write(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)

def save_jobs():
    _atomic_write(STATE_FILE, {"jobs": jobs})

def load_jobs():
    if STATE_FILE.is_file():
        try:
            data = json.loads(STATE_FILE.read_text())
            jobs.update({k: v for k, v in data.get("jobs", {}).items()})
            # clean transitional states after a crash/restart
            for jid, j in jobs.items():
                # anything not terminal or queued → make it queued
                if j.get("status") not in ("completed", "failed", "cancelled", "queued"):
                    j["status"] = "queued"
                    j["stage"] = "queued"
                if j.get("status") == "queued":
                    j["progress"] = 0.0
                    j["current_step"] = "Queued"
                    j["error"] = None
                    j["started_at"] = None
                    j["completed_at"] = None
            save_jobs()

        except Exception as e:
            print(f"[Qwen-Studio] Failed to load jobs.json: {e}")

def cancel_inproc_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return
    # mark cancelled
    job["status"] = "cancelled"
    job["stage"] = "cancelled"
    job["current_step"] = "Cancelled"
    # mark active stages as cancelled
    for s, st in job.get("stages", {}).items():
        if st.get("status") == "active":
            st["status"] = "cancelled"
            st["progress"] = st.get("progress", 0.0)
    job["completed_at"] = now_iso()
    save_jobs()


jobs: Dict[str, dict] = {}
job_queue: List[str] = []

# --- websocket hub
class Hub:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def remove(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        data = json.dumps(message)
        for ws in self.active:
            try:
                if ws.application_state == WebSocketState.CONNECTED:
                    await ws.send_text(data)
                else:
                    dead.append(ws)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove(ws)

hub = Hub()

def stage_update(job: dict, stage: str, note: str = "", pct: float | None = None):
    if stage == "model_loading":
        job["stage"] = "loading_model"
        job["current_step"] = note or "Loading model"
        job["stages"]["model_loading"]["status"] = "active"
        if pct is not None:
            job["stages"]["model_loading"]["progress"] = max(job["stages"]["model_loading"]["progress"], pct)
    elif stage == "pipeline_loading":
        job["stages"]["model_loading"]["status"] = "completed"
        job["stages"]["model_loading"]["progress"] = 1.0
        job["stage"] = "loading_pipeline"
        job["current_step"] = note or "Loading pipeline"
        job["stages"]["pipeline_loading"]["status"] = "active"
        if pct is not None:
            job["stages"]["pipeline_loading"]["progress"] = max(job["stages"]["pipeline_loading"]["progress"], pct)
    elif stage == "lora_loading" and "lora_loading" in job["stages"]:
        for s in ("model_loading", "pipeline_loading"):
            if job["stages"].get(s, {}).get("status") == "active":
                job["stages"][s]["status"] = "completed"; job["stages"][s]["progress"] = 1.0
        job["stage"] = "lora_loading"
        job["current_step"] = note or "Merging LoRA"
        job["stages"]["lora_loading"]["status"] = "active"
        if pct is not None:
            job["stages"]["lora_loading"]["progress"] = max(job["stages"]["lora_loading"]["progress"], pct)
    elif stage == "generation":
        for s in ("model_loading", "pipeline_loading", "lora_loading"):
            if s in job["stages"] and job["stages"][s]["status"] == "active":
                job["stages"][s]["status"] = "completed"; job["stages"][s]["progress"] = 1.0
        job["stage"] = "generation"
        job["current_step"] = note or "Generating"
        job["stages"]["generation"]["status"] = "active"
        if pct is not None:
            job["stages"]["generation"]["progress"] = max(job["stages"]["generation"]["progress"], pct)
            job["progress"] = job["stages"]["generation"]["progress"]

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# --- gpu stats helpers
class GPUMonitor:
    def __init__(self):
        self.stats = {
            "gpu_utilization": 0,
            "vram_used": 0,
            "vram_total": 0,
            "vram_used_percent": 0,
            "gtt_used": 0,
            "gtt_total": 0,
            "gtt_used_percent": 0,
            "gpu_temperature": 0,
            "gpu_name": "",
            "last_update": 0
        }
        self.rocm_smi_path = self.find_rocm_smi()
        
    def find_rocm_smi(self):
        return 'rocm-smi'
    
    def get_stats(self):
        if not self.rocm_smi_path:
            return self.stats
        try:
            # GPU utilization
            result = subprocess.run([self.rocm_smi_path, '--showuse', '--json'],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'card0' in data and 'GPU use (%)' in data['card0']:
                    use = str(data['card0']['GPU use (%)']).replace('%', '')
                    self.stats["gpu_utilization"] = int(float(use))
            
            # VRAM info
            result = subprocess.run([self.rocm_smi_path, '--showmeminfo', 'vram', '--json'],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'card0' in data:
                    card = data['card0']
                    if 'VRAM Total Memory (B)' in card and 'VRAM Total Used Memory (B)' in card:
                        total_b = int(card['VRAM Total Memory (B)'])
                        used_b  = int(card['VRAM Total Used Memory (B)'])
                        total_mb = total_b // (1024 * 1024)
                        used_mb  = used_b  // (1024 * 1024)
                        self.stats["vram_total"] = total_mb
                        self.stats["vram_used"] = used_mb
                        self.stats["vram_used_percent"] = int((used_mb / total_mb) * 100) if total_mb else 0
            
            # GTT info
            result = subprocess.run([self.rocm_smi_path, '--showmeminfo', 'gtt', '--json'],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'card0' in data:
                    card = data['card0']
                    if 'GTT Total Memory (B)' in card and 'GTT Total Used Memory (B)' in card:
                        total_b = int(card['GTT Total Memory (B)'])
                        used_b  = int(card['GTT Total Used Memory (B)'])
                        total_mb = total_b // (1024 * 1024)
                        used_mb  = used_b  // (1024 * 1024)
                        self.stats["gtt_total"] = total_mb
                        self.stats["gtt_used"] = used_mb
                        self.stats["gtt_used_percent"] = int((used_mb / total_mb) * 100) if total_mb else 0
            
            # Temperature
            result = subprocess.run([self.rocm_smi_path, '--showtemp', '--json'],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'card0' in data:
                    card = data['card0']
                    if 'Temperature (Sensor edge) (C)' in card:
                        temp = str(card['Temperature (Sensor edge) (C)']).replace('°C', '').strip()
                        self.stats["gpu_temperature"] = int(float(temp))
            
            # GPU Name (only if empty)
            if not self.stats["gpu_name"]:
                result = subprocess.run([self.rocm_smi_path, '--showproductname', '--json'],
                                        capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if 'card0' in data:
                        card = data['card0']
                        if 'Card Series' in card and str(card['Card Series']).strip() not in ('N/A', ''):
                            self.stats["gpu_name"] = str(card['Card Series']).strip()
                        elif 'GFX Version' in card:
                            self.stats["gpu_name"] = str(card['GFX Version']).strip()
        except Exception as e:
            print(f"GPU monitoring error: {e}")
        
        self.stats["last_update"] = time.time()
        return self.stats

gpu_monitor = GPUMonitor()

async def annotate_png_with_command(path: Path, cmd_str: str):
    try:
        from PIL import Image, PngImagePlugin
        if path.suffix.lower() != ".png":
            return
        img = Image.open(path)
        meta = PngImagePlugin.PngInfo()
        for k, v in img.info.items():
            try:
                meta.add_text(str(k), str(v))
            except Exception:
                pass
        meta.add_text("qwen_image_studio_command", cmd_str)
        meta.add_text("qwen_image_studio_timestamp", datetime.utcnow().isoformat())
        tmp = path.with_suffix(".tmp.png")
        img.save(tmp, "PNG", pnginfo=meta)
        img.close()
        tmp.replace(path)
    except Exception:
        pass

async def process_queue():
    while True:
        try:
            job_id = job_queue.pop(0)
        except asyncio.QueueEmpty:
            await asyncio.sleep(1)
            continue

        job = jobs.get(job_id)
        if not job or job["status"] != "queued":
            continue

        # mark running
        job["status"] = "running"
        job["started_at"] = now_iso()
        job["stage"] = "starting"
        save_jobs()

        # in-proc path (no subprocess)
        job["command"] = "(in-proc) qwen_image_mps.core"
        save_jobs()

        try:
            def _cb(stage, msg, p):
                stage_update(job, stage, msg or "", p)
                asyncio.create_task(hub.broadcast({"type": "job_update", "job": job}))

            params = job["params"]
            saved_paths = []

            if job["type"] == "generate":
                # show “instant” stage completion when cached; core will emit progress if not
                stage_update(job, "model_loading", "Ensuring model", None)
                stage_update(job, "pipeline_loading", "Ensuring pipeline", None)
                if params.get("fast") or params.get("ultra_fast"):
                    job["stages"].setdefault("lora_loading", {"label":"LoRA","status":"pending","progress":0.0})
                    stage_update(job, "lora_loading", "Ensuring LoRA", None)

                paths = await asyncio.to_thread(
                    manager.generate,
                    params["prompt"],
                    steps=params.get("steps", 50),
                    seed=params.get("seed"),
                    num_images=params.get("num_images", 1),
                    size=params.get("size", "16:9"),
                    fast=bool(params.get("fast")),
                    ultra_fast=bool(params.get("ultra_fast")),
                    lora=params.get("lora"),
                    batman=bool(params.get("batman")),
                    cb=_cb,
                )
                saved_paths.extend(paths)

            else:  # edit
                stage_update(job, "model_loading", "Ensuring model", None)
                stage_update(job, "pipeline_loading", "Ensuring pipeline", None)
                if params.get("fast") or params.get("ultra_fast"):
                    job["stages"].setdefault("lora_loading", {"label":"LoRA","status":"pending","progress":0.0})
                    stage_update(job, "lora_loading", "Ensuring LoRA", None)

                img_arg = params["image_path"]
                if not Path(img_arg).is_absolute():
                    img_arg = str((STATE_DIR / img_arg).resolve())

                paths = await asyncio.to_thread(
                    manager.edit,
                    image_path=img_arg,
                    prompt=params["prompt"],
                    steps=params.get("steps", 50),
                    seed=params.get("seed"),
                    fast=bool(params.get("fast")),
                    ultra_fast=bool(params.get("ultra_fast")),
                    lora=params.get("lora"),
                    batman=bool(params.get("batman")),
                    output=params.get("output"),
                    cb=_cb,
                )
                saved_paths.extend(paths)

            # complete + move/annotate
            job["status"] = "completed"
            job["stage"] = "completed"
            for s in job["stages"]:
                if job["stages"][s]["status"] == "active":
                    job["stages"][s]["status"] = "completed"
                    job["stages"][s]["progress"] = 1.0
            job["completed_at"] = now_iso()

            out_dir = STATE_DIR / "jobs" / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            moved = []
            for p in saved_paths:
                src = Path(p)
                dst = out_dir / src.name
                try:
                    shutil.move(str(src), str(dst))
                except Exception:
                    shutil.copy2(str(src), str(dst))
                moved.append(f"jobs/{job_id}/{src.name}")
            job["outputs"] = moved

            for rel in job["outputs"]:
                await annotate_png_with_command(STATE_DIR / rel, job["command"])

            save_jobs()

        except Exception as e:
            job["status"] = "failed"
            job["stage"] = "failed"
            job["error"] = f"{type(e).__name__}: {e}"
            save_jobs()

def new_job(type_: str, params: dict, max_retries: int) -> dict:
    jid = str(uuid.uuid4())
    stages = {
        "model_loading": {"label": "Model", "status": "pending", "progress": 0.0},
        "pipeline_loading": {"label": "Pipeline", "status": "pending", "progress": 0.0},
        "generation": {"label": "Generation", "status": "pending", "progress": 0.0},
    }
    if params.get("fast") or params.get("ultra_fast"):
        stages["lora_loading"] = {"label": "LoRA", "status": "pending", "progress": 0.0}

    job = {
        "id": jid,
        "type": type_,
        "params": params,
        "status": "queued",
        "stage": "queued",
        "stages": stages,
        "progress": 0.0,
        "created_at": now_iso(),
        "started_at": None,
        "completed_at": None,
        "retry_count": 0,
        "max_retries": max(0, int(max_retries)),
        "command": "",
        "outputs": [],
    }
    jobs[jid] = job
    job_queue.append(jid)
    save_jobs()
    return job

@app.get("/api/jobs")
async def api_jobs():
    return {"jobs": list(jobs.values())}

from fastapi import HTTPException

@app.delete("/api/jobs/{job_id}")
async def api_delete_job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        return {"ok": True}

    # if queued, remove from queue
    if j.get("status") == "queued":
        try:
            job_queue.remove(job_id)
        except ValueError:
            pass

    # if running, no subprocess to kill (in-proc path)
    if job_id in running_processes:
        running_processes.pop(job_id, None)

    # remove from dict and disk
    jobs.pop(job_id, None)
    try:
        shutil.rmtree(STATE_DIR / "jobs" / job_id, ignore_errors=True)
    except Exception as e:
        print(f"[Qwen-Studio] delete rmtree error {job_id}: {e}")

    save_jobs()
    await hub.broadcast({"type": "job_deleted", "id": job_id})
    return {"ok": True}


@app.get("/api/file")
async def api_file(path: str):
    # Reject absolute inputs outright
    if Path(path).is_absolute():
        raise HTTPException(status_code=403, detail="Forbidden")

    base = STATE_DIR.resolve()
    p = (base / path).resolve()

    # Enforce sandbox
    if base not in p.parents and p != base:
        raise HTTPException(status_code=403, detail="Forbidden")

    if not p.is_file():
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(str(p))


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(PUBLIC_DIR / "index.html")

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await hub.connect(ws)
    try:
        await ws.send_text(json.dumps({"type": "init", "jobs": list(jobs.values())}))
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            
            if data.get("type") == "cancel_job":
                jid = data.get("job_id")
                print(f"[Qwen-Studio] Cancelling job {jid}")
                j = jobs.get(jid)
                if j and j["status"] in ("queued", "running", "processing"):
                    prev_status = j["status"]

                    j["status"] = "cancelled"
                    j["stage"] = "cancelled"
                    j["current_step"] = "Cancelled"
                    j["updated_at"] = now_iso()
                    if j.get("started_at") and not j.get("completed_at"):
                        j["completed_at"] = j["updated_at"]
                    for s in j["stages"].values():
                        if s.get("status") == "active":
                            s["status"] = "completed"
                            s["progress"] = 1.0

                    if prev_status == "queued":
                        try:
                            job_queue.remove(jid)
                        except ValueError:
                            pass
                    else:
                        running_processes.pop(jid, None)

                    await hub.broadcast({"type": "job_update", "job": j})
                    save_jobs()

            
            elif data.get("type") == "restart_job":
                jid = data.get("job_id")
                j = jobs.get(jid)
                if j and j["status"] in ("failed", "cancelled", "completed"):
                    j["status"] = "queued"
                    j["stage"] = "queued"
                    j["progress"] = 0.0
                    j["retry_count"] = 0
                    j["error"] = None
                    for s in j["stages"].values():
                        s["status"] = "pending"
                        s["progress"] = 0.0
                    job_queue.append(jid)
                    await hub.broadcast({"type": "job_update", "job": j})
                    
    except WebSocketDisconnect:
        hub.remove(ws)
        
@app.post("/api/generate")
async def api_generate(
    prompt: str = Form(...),
    fast: bool = Form(False),
    ultra_fast: bool = Form(False),
    steps: Optional[int] = Form(50),
    seed: Optional[int] = Form(None),
    num_images: Optional[int] = Form(1),
    lora: Optional[str] = Form(None),
    batman: Optional[bool] = Form(False),
    size: str = Form("16:9"),    
    max_retries: Optional[int] = Form(3),
):
    params = {
        "prompt": prompt,
        "steps": int(steps) if steps is not None else 50,
        "seed": int(seed) if seed not in (None, "",) else None,
        "num_images": max(1, int(num_images) if num_images else 1),
        "lora": lora or None,
        "batman": bool(batman),
        "fast": bool(fast),
        "ultra_fast": bool(ultra_fast),
        "size": size,  
    }
    job = new_job("generate", params, max_retries=max_retries if max_retries is not None else 3)
    await hub.broadcast({"type": "job_update", "job": job})
    return {"job_id": job["id"]}

@app.post("/api/edit")
async def api_edit(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    fast: bool = Form(False),
    ultra_fast: bool = Form(False),
    steps: Optional[int] = Form(50),
    seed: Optional[int] = Form(None),
    lora: Optional[str] = Form(None),
    batman: Optional[bool] = Form(False),
    size: str = Form("16:9"),    
    max_retries: Optional[int] = Form(3),
):
    suffix = Path(image.filename or "").suffix or ".png"
    with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=suffix, delete=False) as tf:
        shutil.copyfileobj(image.file, tf)
        image_path = str(Path(tf.name).resolve())

    params = {
        "prompt": prompt,
        "image_path": image_path,
        "steps": int(steps) if steps is not None else 50,
        "seed": int(seed) if seed not in (None, "",) else None,
        "lora": lora or None,
        "batman": bool(batman),
        "fast": bool(fast),
        "ultra_fast": bool(ultra_fast),
        "size": size,  
        "output": None,
    }
    job = new_job("edit", params, max_retries=max_retries if max_retries is not None else 3)
    job_dir = STATE_DIR / "jobs" / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(image_path).suffix or ".png"
    dest = job_dir / f"input{ext}"
    try:
        shutil.move(image_path, dest)
    except Exception:
        shutil.copy2(image_path, dest)
    job["params"]["image_path"] = f"jobs/{job['id']}/input{ext}"
    save_jobs()

    await hub.broadcast({"type": "job_update", "job": job})
    return {"job_id": job["id"]}

async def gpu_stats_broadcaster():
    while True:
        try:
            stats = gpu_monitor.get_stats()
            await hub.broadcast({"type": "gpu_stats", "stats": stats})
        except Exception as e:
            print(f"GPU stats broadcast error: {e}")
        await asyncio.sleep(2)

@app.on_event("startup")
async def _startup():
    load_jobs()
    job_queue.clear()
    job_queue.extend([jid for jid, j in jobs.items() if j.get("status") == "queued"])
    asyncio.create_task(process_queue())
    asyncio.create_task(gpu_stats_broadcaster())
