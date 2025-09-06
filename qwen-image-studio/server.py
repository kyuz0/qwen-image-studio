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
import subprocess
import json as _json
import time
import subprocess
import shlex

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

# CLI path from same repo; override with env QIM_CLI_PATH if needed
CLI_PATH = Path(os.getenv("QIM_CLI_PATH") or (PROJECT_ROOT / "qwen-image-mps.py")).resolve()
PYTHON_BIN = os.getenv("QIM_PYTHON_BIN", "python")

# uploads
UPLOAD_DIR = Path(tempfile.gettempdir()) / "qwen-image-studio"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

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

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def build_command(job: dict) -> List[str]:
    t = job["type"]
    p = job["params"]
    cmd = [PYTHON_BIN, str(CLI_PATH)]
    if t == "generate":
        cmd += ["generate", "-p", p["prompt"]]
        if p.get("steps"):
            cmd += ["--steps", str(p["steps"])]
        if p.get("seed") is not None:
            cmd += ["--seed", str(p["seed"])]
        if p.get("num_images"):
            cmd += ["--num-images", str(p["num_images"])]
        if p.get("size"):
            cmd += ["--size", p["size"]]
    else:
        img_arg = p["image_path"]
        if not Path(img_arg).is_absolute():
            img_arg = str((STATE_DIR / img_arg).resolve())
        cmd += ["edit", "-i", img_arg, "-p", p["prompt"]]
        if p.get("steps"):
            cmd += ["--steps", str(p["steps"])]
        if p.get("seed") is not None:
            cmd += ["--seed", str(p["seed"])]
        if p.get("output"):
            cmd += ["--output", p["output"]]
    if p.get("fast"):
        cmd += ["--fast"]
    if p.get("ultra_fast"):
        cmd += ["--ultra-fast"]
    if p.get("lora"):
        cmd += ["--lora", p["lora"]]
    if p.get("batman"):
        cmd += ["--batman"]
    return cmd


def cmd_to_string(cmd: List[str]) -> str:
    def q(s: str) -> str:
        return s if re.fullmatch(r"[A-Za-z0-9_\-./:]+", s) else json.dumps(s)
    return " ".join(q(x) for x in cmd)

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


# --- progress helpers
TQDM_RE = re.compile(r"(\d+)%\|")
SAVE_SINGLE_RE = re.compile(r"Image saved to:\s+(?P<path>.+)")
SAVE_EDIT_RE = re.compile(r"Edited image saved to:\s+(?P<path>.+)")

def extract_progress(line: str):
    # 1) Plain percent anywhere: "67%" or " 67%|"
    m = re.search(r'(\d{1,3})\s*%(?:\s|[\]|])', line)
    if m:
        pct = int(m.group(1))
        if 0 <= pct <= 100:
            return pct / 100.0

    # 2) tqdm style counters: "3/5 [" or "3/5["
    m = re.search(r'(\d+)\s*/\s*(\d+)\s*\[', line)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den > 0:
            return max(0.0, min(1.0, num / den))

    # 3) Some lines imply completion
    if "CLI: Denoising finished" in line or "CLI: Pipeline ready on device" in line:
        return 1.0

    return None


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

async def iter_lines_preserve_cr(stream):
    """
    Read from an async stream while echoing raw output. 
    Now yields on both \\n AND \\r to catch tqdm progress updates.
    """
    buf = ""
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="ignore")

        # Echo EXACTLY as received so CR-based progress stays on one line
        sys.stdout.write(text)
        sys.stdout.flush()

        buf += text
        while True:
            # Find the next delimiter (newline or carriage return)
            n_pos = buf.find("\n")
            r_pos = buf.find("\r")
            
            if n_pos == -1 and r_pos == -1:
                break
            
            # Use whichever comes first
            if n_pos == -1:
                delimiter_pos = r_pos
                delimiter = "\r"
            elif r_pos == -1:
                delimiter_pos = n_pos  
                delimiter = "\n"
            else:
                delimiter_pos = min(n_pos, r_pos)
                delimiter = "\r" if r_pos < n_pos else "\n"
            
            line = buf[:delimiter_pos].rstrip("\r\n")
            buf = buf[delimiter_pos + 1:]
            
            # Only yield non-empty lines to avoid spam
            if line.strip():
                yield line
                
    if buf.strip():
        # emit any trailing partial line once the process ends
        yield buf.rstrip("\r\n")

async def process_queue():
    while True:
        if not job_queue:
            await asyncio.sleep(0.1)
            continue
        job_id = job_queue.pop(0)
        job = jobs.get(job_id)
        if not job or job["status"] != "queued":
            continue
        job["status"] = "processing"
        job["stage"] = "loading_model"
        job["stages"]["model_loading"]["status"] = "active"
        job["started_at"] = now_iso()
        await hub.broadcast({"type": "job_update", "job": job})
        save_jobs()

        cmd = build_command(job)
        cmd_str = cmd_to_string(cmd)
        job["command"] = cmd_str

        print(f"[Qwen-Studio] Executing command: {cmd_str}", flush=True)

        current_stage = "model_loading"
        saved_paths: List[str] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            running_processes[job_id] = proc 
            assert proc.stdout is not None
            async for line in iter_lines_preserve_cr(proc.stdout):
                line = line.rstrip()
                
                # stages (guard against regression)
                if "Loading checkpoint shards" in line:
                    if current_stage in ("lora_loading", "generation"):
                        pass
                    else:
                        if current_stage != "model_loading":
                            current_stage = "model_loading"
                            job["stage"] = "loading_model"
                            job["current_step"] = "Loading model weights"
                            job["stages"]["model_loading"]["status"] = "active"
                        prog = extract_progress(line)
                        if prog is not None:
                            job["stages"]["model_loading"]["progress"] = max(
                                job["stages"]["model_loading"]["progress"], prog
                            )
                        await hub.broadcast({"type": "job_update", "job": job})

                elif "Loading pipeline components" in line:
                    if current_stage not in ("lora_loading", "generation"):
                        if current_stage == "model_loading":
                            job["stages"]["model_loading"]["status"] = "completed"
                            job["stages"]["model_loading"]["progress"] = 1.0
                        if current_stage != "pipeline_loading":
                            current_stage = "pipeline_loading"
                            job["stage"] = "loading_pipeline"
                            job["current_step"] = "Loading pipeline components"
                            job["stages"]["pipeline_loading"]["status"] = "active"
                        prog = extract_progress(line)
                        if prog is not None:
                            job["stages"]["pipeline_loading"]["progress"] = max(
                                job["stages"]["pipeline_loading"]["progress"], prog
                            )
                        await hub.broadcast({"type": "job_update", "job": job})
                # LoRA merging (tqdm lines)
                elif "LoRA merge:" in line or "Merging LoRA:" in line:
                    if "lora_loading" in job["stages"]:
                        # flip into LoRA stage if we aren't already
                        if current_stage in ("model_loading", "pipeline_loading"):
                            job["stages"][current_stage]["status"] = "completed"
                            job["stages"][current_stage]["progress"] = 1.0
                        if current_stage != "lora_loading":
                            current_stage = "lora_loading"
                            job["stage"] = "lora_loading"
                            job["current_step"] = "Merging LoRA"
                            job["stages"]["lora_loading"]["status"] = "active"
                    # drive LoRA progress with tqdm percent or N/N
                    prog = extract_progress(line)
                    if prog is not None:
                        job["stages"]["lora_loading"]["progress"] = max(
                            job["stages"]["lora_loading"]["progress"], prog
                        )
                    await hub.broadcast({"type": "job_update", "job": job})
                elif "Lightning LoRA" in line or "Merged" in line:
                    if "lora_loading" in job["stages"]:
                        if current_stage in ("model_loading", "pipeline_loading"):
                            job["stages"][current_stage]["status"] = "completed"
                            job["stages"][current_stage]["progress"] = 1.0
                        current_stage = "lora_loading"
                        job["stage"] = "lora_loading"
                        job["current_step"] = "Loading Lightning LoRA"
                        job["stages"]["lora_loading"]["status"] = "active"
                        await hub.broadcast({"type": "job_update", "job": job})

                if ("Denoising started" in line) or ("steps, CFG scale" in line) or ("Editing config:" in line) or ("Generation config:" in line):
                    for s in ("model_loading", "pipeline_loading", "lora_loading"):
                        if s in job["stages"] and job["stages"][s]["status"] == "active":
                            job["stages"][s]["status"] = "completed"
                            job["stages"][s]["progress"] = 1.0
                    current_stage = "generation"
                    job["stage"] = "generation"
                    job["current_step"] = "Generating"
                    job["stages"]["generation"]["status"] = "active"
                    await hub.broadcast({"type": "job_update", "job": job})

                prog = extract_progress(line)
                if prog is not None and current_stage == "generation":
                    job["stages"]["generation"]["progress"] = max(
                        job["stages"]["generation"]["progress"], prog
                    )
                    job["progress"] = job["stages"]["generation"]["progress"]
                    await hub.broadcast({"type": "job_update", "job": job})

                m1 = SAVE_SINGLE_RE.search(line)
                if m1:
                    saved_paths.append(m1.group("path").strip())
                m2 = SAVE_EDIT_RE.search(line)
                if m2:
                    saved_paths.append(m2.group("path").strip())

            ret = await proc.wait()
            running_processes.pop(job_id, None)
            if job["status"] == "cancelled":
                await hub.broadcast({"type": "job_update", "job": job})
                continue
            elif ret != 0:
                job["status"] = "failed"
                job["stage"] = "failed"
                job["error"] = f"Process exited with code {ret}"
            else:
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

        # retry policy
        if job["status"] == "failed":
            job["retry_count"] += 1
            if job["retry_count"] <= job["max_retries"]:
                job["status"] = "queued"
                job["stage"] = "queued"  
                job["progress"] = 0.0
                job["error"] = None  # Clear error on retry
                # Reset non-completed stages for retry
                for s in job["stages"].values():
                    if s["status"] != "completed":
                        s["status"] = "pending"
                        s["progress"] = 0.0
                job_queue.insert(0, job_id)

        await hub.broadcast({"type": "job_update", "job": job})
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
        try: job_queue.remove(job_id)
        except ValueError: pass

    # if running, terminate
    if job_id in running_processes:
        proc = running_processes.pop(job_id)
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                proc.kill()
        except Exception as e:
            print(f"[Qwen-Studio] delete terminate error {job_id}: {e}")

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
                if j and j["status"] in ("queued", "processing"):
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
                        if jid in running_processes:
                            proc = running_processes[jid]
                            try:
                                proc.terminate()
                                print(f"[Qwen-Studio] Terminated process for job {jid}")
                                try:
                                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                                except asyncio.TimeoutError:
                                    proc.kill()
                                    print(f"[Qwen-Studio] Force killed process for job {jid}")
                            except Exception as e:
                                print(f"[Qwen-Studio] Error terminating process {jid}: {e}")
                            finally:
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
