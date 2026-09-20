"""H3 Client — Interface between AudioReactiveController (ARC) and ComfyUI MiniMax H3.

Handles workflow JSON parameterization, prompt queuing, status polling,
and automatic output retrieval into ARC's clip folders.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

DEFAULT_COMFY_URL = os.getenv("COMFYUI_SERVER_ADDRESS", "http://127.0.0.1:8188")
DEFAULT_COMFY_OUTPUT = os.getenv("COMFYUI_OUTPUT_DIR", r"F:\AppsCrucial\ComfyUI_phoenix3\ComfyUI\output")

# Standard workflow search directories
WORKFLOW_SEARCH_DIRS = [
    Path(__file__).resolve().parent.parent.parent / "workflows",
    Path(r"F:\workspace\telegram-server\workflows"),
]


def find_workflow_path(filename: str) -> Optional[Path]:
    """Search for workflow JSON in configured directories."""
    for base in WORKFLOW_SEARCH_DIRS:
        cand = base / filename
        if cand.is_file():
            return cand
    return None


class ComfyH3Client:
    """Client for dispatching and monitoring MiniMax H3 jobs in ComfyUI."""

    def __init__(
        self,
        server_url: str = DEFAULT_COMFY_URL,
        comfy_output_dir: str = DEFAULT_COMFY_OUTPUT,
    ):
        self.server_url = server_url.rstrip("/")
        self.comfy_output_dir = Path(comfy_output_dir)

    def is_available(self, timeout: float = 3.0) -> bool:
        """Check if ComfyUI server is reachable."""
        try:
            req = urllib.request.Request(f"{self.server_url}/system_stats")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.server_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ComfyUI HTTP {e.code} on {endpoint}: {err_msg}")

    def _get(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.server_url}{endpoint}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ComfyUI HTTP {e.code} on {endpoint}: {err_msg}")

    def upload_image(self, image_path: Path) -> str:
        """Upload an image to ComfyUI input directory via HTTP multipart form."""
        url = f"{self.server_url}/upload/image"
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        filename = image_path.name
        
        with open(image_path, "rb") as f:
            content = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("name", filename)

    def queue_prompt(self, workflow_graph: Dict[str, Any]) -> str:
        """Queue a ComfyUI graph and return prompt_id."""
        client_id = str(uuid.uuid4())
        res = self._post("/prompt", {"prompt": workflow_graph, "client_id": client_id})
        prompt_id = res.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"Failed to obtain prompt_id from ComfyUI: {res}")
        return prompt_id

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout_s: float = 600.0,
        poll_interval: float = 2.0,
    ) -> List[str]:
        """Poll /history until the job completes and return output video paths."""
        start_time = time.time()
        while time.time() - start_time < timeout_s:
            try:
                hist = self._get(f"/history/{prompt_id}")
            except Exception:
                hist = {}

            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI job failed: {messages}")

                outputs = hist[prompt_id].get("outputs", {})
                files = []
                for node_output in outputs.values():
                    # Check for video, gifs, or images
                    for category in ("video", "gifs", "images"):
                        for item in node_output.get(category, []):
                            if isinstance(item, dict) and item.get("filename"):
                                sub = item.get("subfolder", "")
                                rel = os.path.join(sub, item["filename"]).replace("\\", "/")
                                files.append(rel)
                if files:
                    return files

            time.sleep(poll_interval)

        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {timeout_s}s")

    def retrieve_output(self, rel_output_path: str, destination_path: Path) -> Path:
        """Copy the generated video from ComfyUI output directory to destination."""
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        local_src = self.comfy_output_dir / rel_output_path
        if local_src.is_file():
            shutil.copy2(local_src, destination_path)
            return destination_path

        # Fallback to HTTP download if local path does not exist
        filename = os.path.basename(rel_output_path)
        subfolder = os.path.dirname(rel_output_path)
        query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
        download_url = f"{self.server_url}/view?{query}"
        urllib.request.urlretrieve(download_url, str(destination_path))
        return destination_path

    def generate_t2v_scene(
        self,
        prompt_text: str,
        duration_sec: float = 13.0,
        seed: Optional[int] = None,
        workflow_template_path: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Execute a full T2V H3 scene generation and save output."""
        wf_file = Path(workflow_template_path) if workflow_template_path else find_workflow_path("workflow_api_h3_t2v_turbo.json")
        if not wf_file or not wf_file.is_file():
            raise FileNotFoundError(f"H3 T2V workflow not found: {workflow_template_path or 'workflow_api_h3_t2v_turbo.json'}")

        with open(wf_file, "r", encoding="utf-8") as f:
            graph = json.load(f)

        # Inject prompt into node 105:104 (or find PrimitiveString/prompt node)
        prompt_node = graph.get("105:104") or next(
            (v for v in graph.values() if v.get("class_type") in ("PrimitiveString", "CLIPTextEncode") and "prompt" in v.get("inputs", {})),
            None,
        )
        if prompt_node:
            prompt_node["inputs"]["prompt"] = prompt_text

        # Inject duration into seconds node 105:111
        sec_node = graph.get("105:111")
        if sec_node and "value" in sec_node.get("inputs", {}):
            sec_node["inputs"]["value"] = float(duration_sec)

        # Inject noise seed into node 105:15
        if seed is not None:
            seed_node = graph.get("105:15")
            if seed_node and "noise_seed" in seed_node.get("inputs", {}):
                seed_node["inputs"]["noise_seed"] = int(seed)

        pid = self.queue_prompt(graph)
        outputs = self.wait_for_completion(pid)
        if not outputs:
            raise RuntimeError(f"ComfyUI completed prompt {pid} but produced no output files")

        dest = output_path or Path(f"render_output/h3_{pid[:8]}.mp4")
        return self.retrieve_output(outputs[0], dest)

    def generate_flf_transition(
        self,
        img1_path: Path,
        img2_path: Path,
        prompt_text: str,
        duration_sec: float = 11.54,
        seed: Optional[int] = None,
        workflow_template_path: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Execute a First-and-Last-Frame morphing transition between two images."""
        wf_file = Path(workflow_template_path) if workflow_template_path else find_workflow_path("workflow_api_h3_flf_turbo.json")
        if not wf_file or not wf_file.is_file():
            raise FileNotFoundError(f"H3 FLF workflow not found: {workflow_template_path or 'workflow_api_h3_flf_turbo.json'}")

        with open(wf_file, "r", encoding="utf-8") as f:
            graph = json.load(f)

        # Upload images if needed
        up1 = self.upload_image(img1_path)
        up2 = self.upload_image(img2_path)

        # Inject image inputs (nodes 114 and 116 in standard H3 FLF)
        if "114" in graph and "inputs" in graph["114"]:
            graph["114"]["inputs"]["image"] = up1
        if "116" in graph and "inputs" in graph["116"]:
            graph["116"]["inputs"]["image"] = up2

        # Inject prompt & duration
        if "105:104" in graph:
            graph["105:104"]["inputs"]["prompt"] = prompt_text
        if "105:111" in graph:
            graph["105:111"]["inputs"]["value"] = float(duration_sec)
        if seed is not None and "105:15" in graph:
            graph["105:15"]["inputs"]["noise_seed"] = int(seed)

        pid = self.queue_prompt(graph)
        outputs = self.wait_for_completion(pid)
        if not outputs:
            raise RuntimeError(f"ComfyUI completed prompt {pid} but produced no output files")

        dest = output_path or Path(f"render_output/h3_flf_{pid[:8]}.mp4")
        return self.retrieve_output(outputs[0], dest)
