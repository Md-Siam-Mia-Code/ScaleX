# webui_scalex.py
# Apply patches FIRST!
try:
    import patches

    patches.apply_torchvision_patches()
except ImportError:
    print("ScaleX WebUI WARNING: patches.py not found. Proceeding without patches.")
except Exception as e:
    print(f"ScaleX WebUI WARNING: Failed to apply patches: {e}. Proceeding.")

import warnings

warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    message=".*You are using `torch.load` with `weights_only=False`.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.models._utils"
)

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import threading
import time
import uuid
from pathlib import Path
import traceback
import cv2
import numpy as np
import io
from contextlib import redirect_stdout
import re
import platform
import psutil
import shutil
import json


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scalex"))

from inference_scalex import (
    SCALEX_MODEL_CONFIGS,
    REALESRGAN_MODELS,
    FaceModelEnum,
    BGModelEnum,
    get_scalex_model_display_name_and_path,
)
from basicsr.utils import imwrite
from basicsr.utils.download_util import load_file_from_url
from scalex.utils import ScaleXEnhancer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch

APP_VERSION = "N/A"
try:
    with open(PROJECT_ROOT / "VERSION", "r") as f:
        APP_VERSION = f.read().strip()
except FileNotFoundError:
    print("ScaleX WebUI WARNING: VERSION file not found in root directory.")
except Exception as e:
    print(f"ScaleX WebUI WARNING: Could not read VERSION file: {e}")


CONFIG_FILE_PATH = PROJECT_ROOT / "webui_config.json"


def load_app_config():
    defaults = {"output_folder": str(PROJECT_ROOT / "Output" / "webui_output")}
    if CONFIG_FILE_PATH.exists():
        try:
            with open(CONFIG_FILE_PATH, "r") as f:
                user_config = json.load(f)
            if "output_folder" in user_config and user_config["output_folder"]:
                resolved_path = str(Path(user_config["output_folder"]).resolve())
                defaults["output_folder"] = resolved_path
        except Exception as e:
            print(
                f"Warning: Could not load {CONFIG_FILE_PATH}, using defaults. Error: {e}"
            )
    defaults["output_folder"] = str(
        Path(defaults["output_folder"]).resolve()
    )  # Ensure absolute
    return defaults


def save_app_config(new_config_data):
    try:
        current_config = load_app_config()
        if "output_folder" in new_config_data and new_config_data["output_folder"]:
            new_config_data["output_folder"] = str(
                Path(new_config_data["output_folder"]).resolve()
            )
        elif (
            "output_folder" in new_config_data and not new_config_data["output_folder"]
        ):  # Revert to default if empty
            new_config_data["output_folder"] = str(
                Path(PROJECT_ROOT / "Output" / "webui_output").resolve()
            )

        current_config.update(new_config_data)
        with open(CONFIG_FILE_PATH, "w") as f:
            json.dump(current_config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving to {CONFIG_FILE_PATH}: {e}")
        return False


app_runtime_config = load_app_config()

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static",
    template_folder="static",
)

app.config["UPLOAD_FOLDER"] = str(PROJECT_ROOT / "Input" / "webui_input")
app.config["OUTPUT_FOLDER"] = app_runtime_config["output_folder"]

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
app.secret_key = os.urandom(24)

active_tasks = {}


class WebTileProgressStream(io.StringIO):
    def __init__(self, task_id_for_update: str, image_name: str):
        super().__init__()
        self.task_id = task_id_for_update
        self.image_name = image_name
        self.line_buffer = ""
        self.tile_regex = re.compile(r"Tile\s+(\d+)/(\d+)")

    def write(self, s: str):
        global active_tasks
        self.line_buffer += s
        terminators = ["\n", "\r"]
        while any(t in self.line_buffer for t in terminators):
            first_terminator_pos = -1
            used_terminator_len = 0
            for t_char in terminators:
                pos = self.line_buffer.find(t_char)
                if pos != -1 and (
                    first_terminator_pos == -1 or pos < first_terminator_pos
                ):
                    first_terminator_pos = pos
                    used_terminator_len = len(t_char)
            if first_terminator_pos == -1:
                break
            line_to_process = self.line_buffer[:first_terminator_pos]
            self.line_buffer = self.line_buffer[
                first_terminator_pos + used_terminator_len :
            ]
            if line_to_process.strip():
                match = self.tile_regex.search(line_to_process.strip())
                if match and self.task_id in active_tasks:
                    task = active_tasks[self.task_id]
                    current_tile, total_tiles = int(match.group(1)), int(match.group(2))
                    tile_percentage_span = (
                        (current_tile / total_tiles) * 55 if total_tiles > 0 else 0
                    )
                    task["progress"] = min(30 + tile_percentage_span, 85)
                    task["tile_progress"] = {
                        "current": current_tile,
                        "total": total_tiles,
                        "percentage": (
                            (current_tile / total_tiles) * 100 if total_tiles > 0 else 0
                        ),
                    }
                    task["status"] = f"bg_tiling_{current_tile}_of_{total_tiles}"
                    task["current_step_description"] = (
                        f"Background Tiling: Tile {current_tile} of {total_tiles}"
                    )
                    log_to_task(
                        self.task_id, f"Tile: {current_tile}/{total_tiles}", "DEBUG"
                    )
        return len(s)

    def flush(self):
        pass


def log_to_task(task_id: str, message: str, level: str = "INFO"):
    global active_tasks
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] [{level.upper()}] {message}"
    if task_id in active_tasks:
        active_tasks[task_id].setdefault("logs", []).append(log_entry)


def webui_scalex_progress_callback_handler(task_id: str, event_data: dict):
    global active_tasks
    if task_id not in active_tasks:
        return
    task = active_tasks[task_id]
    event_type = event_data.get("event_type")
    current_progress = task.get("progress", 0)
    is_bg_tiling_active = task.get("is_bg_tiling_active", False)
    new_description = task.get("current_step_description", "Processing...")

    if event_type == "face_detection_start":
        new_description = "Detecting faces..."
    elif event_type == "face_detection_done":
        num_faces = event_data.get("num_faces", 0)
        task["num_detected_faces"] = num_faces
        new_description = f"Detected {num_faces} face(s)"
    elif event_type == "face_alignment_start":
        new_description = "Aligning faces..."
    elif event_type == "face_alignment_done":
        num_aligned = event_data.get("num_aligned", 0)
        new_description = f"Aligned {num_aligned} face(s)"
    elif event_type == "no_faces_to_process":
        task["num_detected_faces"] = 0
        new_description = "No faces found to process"
    elif event_type == "processing_face_start":
        current_face = event_data.get("current_face", 1)
        total_faces = event_data.get("total_faces", 1)
        new_description = f"Enhancing face {current_face} of {total_faces}"
    elif event_type == "processing_face_done":
        current_face = event_data.get("current_face", 1)
        total_faces = event_data.get("total_faces", 1)
        new_description = (
            f"All {total_faces} face(s) enhanced"
            if current_face == total_faces
            else task.get("current_step_description")
        )
    elif event_type == "pasting_faces_start":
        new_description = "Pasting faces / Finalizing image..."
    elif event_type in [
        "pasting_faces_done",
        "final_output_ready_aligned",
        "final_outputs_ready_no_paste",
    ]:
        new_description = "Finalizing complete. Preparing to save."
    elif event_type == "bg_upsample_only_start":
        new_description = "Upsampling background (no faces detected)..."
    elif event_type == "bg_upsample_only_done":
        new_description = (
            "Background upsampling complete."
            if event_data.get("success", False)
            else "Background upsampling failed."
        )
    elif event_type == "bg_upsample_for_paste_start":
        new_description = "Upsampling background..."
    elif event_type == "bg_upsample_for_paste_done":
        new_description = (
            "Background upsampling for paste complete."
            if event_data.get("success", False)
            else "Background upsampling for paste failed."
        )

    # Simplified progress update based on event, more refinement might be needed if BG tiling is not active
    if not is_bg_tiling_active:
        progress_map = {
            "face_detection_start": 28,
            "face_detection_done": 30,
            "face_alignment_start": 32,
            "face_alignment_done": 35,
            "no_faces_to_process": 85,
            "processing_face_start": 35,
            "processing_face_done": 80,
            "pasting_faces_start": 85,
            "pasting_faces_done": 88,
            "bg_upsample_only_start": 30,
            "bg_upsample_only_done": 85,
            "bg_upsample_for_paste_start": 82,
            "bg_upsample_for_paste_done": 85,
        }
        if event_type in progress_map:
            current_progress = max(current_progress, progress_map[event_type])
        if (
            event_type == "processing_face_start"
            and event_data.get("total_faces", 0) > 0
        ):
            face_prog = (
                (event_data.get("current_face", 1) - 1)
                / event_data.get("total_faces", 1)
            ) * 45  # 35-80 range
            current_progress = max(current_progress, 35 + face_prog)

    task["progress"] = int(current_progress)
    task["current_step_description"] = new_description
    task["status"] = event_type
    log_to_task(
        task_id,
        f"Event: {event_type} -> {new_description} (Prog: {task['progress']}%)",
        "DEBUG",
    )


def process_image_task_runner(
    task_id: str, params: dict
):  # Largely unchanged, ensure OUTPUT_FOLDER is used correctly from app_runtime_config
    global active_tasks, app_runtime_config
    try:
        active_tasks[task_id]["current_step_description"] = "Initializing task..."
        log_to_task(task_id, "Task starting.")
        input_path_str = params["input_path"]
        output_base_folder = Path(
            app_runtime_config["output_folder"]
        )  # Use current runtime config
        task_output_folder = output_base_folder / task_id
        task_output_folder.mkdir(parents=True, exist_ok=True)
        active_tasks[task_id]["output_subfolder"] = str(task_output_folder)

        # Parameter extraction (same as before)
        face_enhance_model_val = params["face_enhance_model"]
        bg_enhance_model_val = params["bg_enhance_model"]
        overall_upscale = int(params["overall_upscale"])
        bg_tile_size = int(params["bg_tile_size"])
        output_suffix = params.get("output_suffix")
        center_face_only = params.get("center_face_only", "false").lower() == "true"
        aligned_input = params.get("aligned_input", "false").lower() == "true"
        output_ext = params.get("output_ext", "auto")
        device_pref = params.get("device", "auto")
        fidelity_weight_str = params.get("fidelity_weight")
        fidelity_weight = (
            float(fidelity_weight_str)
            if fidelity_weight_str and fidelity_weight_str.strip()
            else None
        )
        save_cropped = params.get("save_cropped", "false").lower() == "true"
        save_restored = params.get("save_restored", "false").lower() == "true"
        save_comparison = params.get("save_comparison", "false").lower() == "true"

        active_tasks[task_id]["progress"] = 5
        active_tasks[task_id][
            "current_step_description"
        ] = "Determining Torch device..."
        torch_device_str = device_pref.lower()
        if torch_device_str == "auto":
            torch_device_str = (
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                    and torch.backends.mps.is_built()
                    else "cpu"
                )
            )
        try:
            selected_torch_device = torch.device(torch_device_str)
        except Exception as e:
            log_to_task(
                task_id,
                f"Error device '{torch_device_str}': {e}. CPU fallback.",
                "WARNING",
            )
            selected_torch_device = torch.device("cpu")
        active_tasks[task_id]["device_used"] = selected_torch_device.type.upper()
        active_tasks[task_id]["progress"] = 10
        active_tasks[task_id]["current_step_description"] = "Loading models..."

        face_model_display_name, final_gfpgan_model_path_str = (
            get_scalex_model_display_name_and_path(face_enhance_model_val)
        )
        bg_upsampler_instance = None
        if bg_enhance_model_val != "none" and bg_enhance_model_val in REALESRGAN_MODELS:
            bg_config = REALESRGAN_MODELS[bg_enhance_model_val]
            active_tasks[task_id][
                "current_step_description"
            ] = f'Loading BG: {bg_config["internal_name"]}...'
            final_bg_model_path = bg_config["model_path"]
            if final_bg_model_path.startswith("https://"):
                dl_dir = PROJECT_ROOT / "models" / "pretrained"
                dl_dir.mkdir(parents=True, exist_ok=True)
                try:
                    final_bg_model_path = load_file_from_url(
                        url=final_bg_model_path,
                        model_dir=str(dl_dir),
                        progress=True,
                        file_name=None,
                    )
                except Exception as e:
                    log_to_task(task_id, f"Error DL BG model: {e}", "ERROR")
                    final_bg_model_path = None
            if final_bg_model_path and Path(final_bg_model_path).is_file():
                try:
                    model_params = bg_config["model_class_params"]
                    realesrgan_model_instance = RRDBNet(**model_params)
                    bg_upsampler_instance = RealESRGANer(
                        scale=bg_config["netscale"],
                        model_path=str(final_bg_model_path),
                        model=realesrgan_model_instance,
                        tile=bg_tile_size,
                        tile_pad=10,
                        pre_pad=0,
                        half=(selected_torch_device.type == "cuda"),
                        device=selected_torch_device,
                    )
                except Exception as e:
                    log_to_task(
                        task_id,
                        f"Could not init BG upsampler: {e}. Disabled.",
                        "WARNING",
                    )
                    bg_upsampler_instance = None

        active_tasks[task_id]["progress"] = 20
        active_tasks[task_id][
            "current_step_description"
        ] = f"Loading ScaleX: {face_model_display_name}..."
        restorer = ScaleXEnhancer(
            model_path=final_gfpgan_model_path_str,
            upscale=float(overall_upscale),
            arch=SCALEX_MODEL_CONFIGS[face_enhance_model_val]["arch"],
            channel_multiplier=SCALEX_MODEL_CONFIGS[face_enhance_model_val][
                "channel_multiplier"
            ],
            bg_upsampler=bg_upsampler_instance,
            device=selected_torch_device,
        )
        active_tasks[task_id]["progress"] = 25
        is_bg_tiling_active_for_task = (
            bg_upsampler_instance is not None
            and isinstance(bg_upsampler_instance, RealESRGANer)
            and bg_tile_size > 0
            and not aligned_input
        )
        active_tasks[task_id]["is_bg_tiling_active"] = is_bg_tiling_active_for_task

        input_image_file_path = Path(input_path_str)
        img_name_stem = input_image_file_path.stem
        active_tasks[task_id]["current_image_name"] = input_image_file_path.name
        active_tasks[task_id][
            "current_step_description"
        ] = f"Reading: {input_image_file_path.name}..."
        input_img_np = cv2.imread(str(input_image_file_path), cv2.IMREAD_COLOR)
        if input_img_np is None:
            raise ValueError(f"Could not read: {input_image_file_path.name}")

        enhance_kwargs = {
            "has_aligned": aligned_input,
            "only_center_face": center_face_only,
            "paste_back": not aligned_input,
            "progress_callback": lambda event_data: webui_scalex_progress_callback_handler(
                task_id, event_data
            ),
        }
        if fidelity_weight is not None:
            enhance_kwargs["weight"] = fidelity_weight

        enh_output_data, enh_exception = None, None
        if is_bg_tiling_active_for_task:
            with redirect_stdout(
                WebTileProgressStream(task_id, input_image_file_path.name)
            ):
                enh_output_data = restorer.enhance(input_img_np, **enhance_kwargs)
        else:
            enh_output_data = restorer.enhance(input_img_np, **enhance_kwargs)
        if enh_exception:
            raise enh_exception  # Should be caught by try-except in enhance if it sets exception

        active_tasks[task_id]["progress"] = 90
        cropped_faces, restored_faces, restored_output_img = enh_output_data
        active_tasks[task_id]["current_step_description"] = "Saving outputs..."
        active_tasks[task_id]["progress"] = 95
        output_ext_final = (
            output_ext.lower()
            if output_ext.lower() != "auto"
            else input_image_file_path.suffix[1:].lower() or "png"
        )

        processed_image_relative_path = None  # For UI display
        saved_file_paths = {}

        def save_image_web(
            img_data, subfolder_name, name_parts_list
        ):  # name_parts_list should be [base, detail, suffix_opt]
            nonlocal processed_image_relative_path
            subfolder_path = task_output_folder / subfolder_name
            subfolder_path.mkdir(exist_ok=True, parents=True)
            base_filename = "_".join(filter(None, name_parts_list))
            filename_with_ext = f"{base_filename}.{output_ext_final}"
            full_save_path = subfolder_path / filename_with_ext
            imwrite(img_data, str(full_save_path))
            relative_path_for_url = Path(task_id) / subfolder_name / filename_with_ext
            if subfolder_name == "restored_imgs" or (
                aligned_input
                and subfolder_name == "aligned_outputs"
                and not processed_image_relative_path
            ):
                processed_image_relative_path = str(relative_path_for_url)
            return str(relative_path_for_url)

        if save_cropped and cropped_faces:
            for i, fn in enumerate(cropped_faces):
                save_image_web(fn, "cropped_faces", [img_name_stem, f"face_{i:02d}"])
        if save_restored and restored_faces:
            for i, fn in enumerate(restored_faces):
                save_image_web(
                    fn,
                    "restored_faces",
                    [img_name_stem, f"face_{i:02d}", output_suffix],
                )
        if (
            save_comparison
            and cropped_faces
            and restored_faces
            and len(cropped_faces) == len(restored_faces)
        ):
            for i, (cf, rf) in enumerate(zip(cropped_faces, restored_faces)):
                try:
                    th, tw = rf.shape[:2]
                    cf_r = cv2.resize(cf, (tw, th)) if cf.shape[:2] != (th, tw) else cf
                    save_image_web(
                        np.concatenate((cf_r, rf), axis=1),
                        "cmp",
                        [img_name_stem, f"cmp_{i:02d}"],
                    )
                except Exception as e:
                    log_to_task(task_id, f"Cmp save fail {i}: {e}", "WARN")

        if not aligned_input and restored_output_img is not None:
            saved_file_paths["main_output"] = save_image_web(
                restored_output_img, "restored_imgs", [img_name_stem, output_suffix]
            )
        elif (
            aligned_input and restored_output_img is not None
        ):  # This is the single restored face
            saved_file_paths["main_output"] = save_image_web(
                restored_output_img, "aligned_outputs", [img_name_stem, output_suffix]
            )

        if processed_image_relative_path:
            active_tasks[task_id]["result_path"] = processed_image_relative_path
        elif (
            restored_faces
        ):  # Fallback for no main image but faces exist (e.g. not paste_back)
            active_tasks[task_id]["result_path"] = save_image_web(
                restored_faces[0],
                "restored_faces",
                [img_name_stem, "face_00", output_suffix, "main"],
            )

        active_tasks[task_id]["status"] = "completed"
        active_tasks[task_id]["progress"] = 100
        active_tasks[task_id]["current_step_description"] = "Processing complete!"
        active_tasks[task_id]["saved_files_info"] = saved_file_paths
        log_to_task(
            task_id,
            f"Result path for UI: {active_tasks[task_id].get('result_path', 'N/A')}",
        )
    except Exception as e:
        tb_str = traceback.format_exc()
        log_to_task(task_id, f"Error: {e}\n{tb_str}", "ERROR")
        if task_id in active_tasks:
            active_tasks[task_id]["status"] = "error"
            active_tasks[task_id]["error"] = str(e)
            active_tasks[task_id]["current_step_description"] = f"Error: {e}"
            active_tasks[task_id]["progress"] = 100
    finally:
        if task_id in active_tasks:
            active_tasks[task_id]["thread_active"] = False


@app.route("/")
def index_route():
    return render_template("index.html", app_version=APP_VERSION)


@app.route("/config_options")
def config_options_route():  # Same as before
    face_models_list = [
        {"value": e.value, "name": SCALEX_MODEL_CONFIGS[e.value]["model_name"]}
        for e in FaceModelEnum
    ]
    bg_models_list = [
        {
            "value": e.value,
            "name": (
                "None"
                if e.value == "none"
                else REALESRGAN_MODELS[e.value]["internal_name"]
            ),
        }
        for e in BGModelEnum
        if e.value == "none" or e.value in REALESRGAN_MODELS
    ]
    return jsonify(
        {
            "face_models": face_models_list,
            "bg_models": bg_models_list,
            "default_face_model": FaceModelEnum.v1_4.value,
            "default_bg_model": BGModelEnum.x4plus.value,
            "default_upscale": 2,
            "default_bg_tile": 400,
            "output_formats": ["auto", "png", "jpg"],
            "devices": ["auto", "cpu", "cuda", "mps"],
        }
    )


@app.route("/process", methods=["POST"])
def process_route():  # Same as before
    global active_tasks
    if "inputFile" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["inputFile"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    form_params = request.form.to_dict()
    original_filename_path = Path(file.filename)
    sanitized_basename = "".join(
        c if c.isalnum() or c in [".", "-", "_"] else "_"
        for c in original_filename_path.stem
    )
    unique_id_for_file = str(uuid.uuid4())[:8]
    sanitized_filename = (
        f"{unique_id_for_file}_{sanitized_basename}{original_filename_path.suffix}"
    )
    upload_path = Path(app.config["UPLOAD_FOLDER"]) / sanitized_filename
    file.save(upload_path)
    task_id = str(uuid.uuid4())
    original_uploaded_relative_path = sanitized_filename
    params_for_task = {**form_params, "input_path": str(upload_path)}
    active_tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "current_step_description": "Waiting in queue...",
        "params_summary": {k: v for k, v in form_params.items() if k != "inputFile"},
        "logs": [
            f"[{time.strftime('%H:%M:%S')}] [INFO] Task {task_id} for {original_filename_path.name}."
        ],
        "thread_active": True,
        "original_filename": original_filename_path.name,
        "original_uploaded_path_for_comparison": original_uploaded_relative_path,
        "tile_progress": None,
        "num_detected_faces": None,
        "is_bg_tiling_active": False,
    }
    thread = threading.Thread(
        target=process_image_task_runner, args=(task_id, params_for_task), daemon=True
    )
    active_tasks[task_id]["thread_obj"] = thread
    thread.start()
    return jsonify(
        {
            "message": "Processing started",
            "task_id": task_id,
            "original_uploaded_path": original_uploaded_relative_path,
            "current_step_description": active_tasks[task_id][
                "current_step_description"
            ],
            "status": active_tasks[task_id]["status"],
        }
    )


@app.route("/uploads/<path:filename>")
def serve_upload_file_route(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/outputs/<path:filepath>")
def serve_output_file_route(filepath):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filepath)


@app.route("/progress/<task_id>")
def progress_status_route(task_id: str):  # Same as before
    global active_tasks
    if task_id not in active_tasks:
        return jsonify({"error": "Invalid task ID"}), 404
    task_info = active_tasks[task_id]
    response_data = {
        k: v
        for k, v in task_info.items()
        if k not in ["thread_obj", "error_traceback", "params_for_task"]
    }
    response_data["logs"] = task_info.get("logs", [])
    if "logs" in task_info:
        task_info["logs"] = []
    return jsonify(response_data)


@app.route("/system_info")
def system_info_route():  # Ensure latest config used
    global app_runtime_config
    app_runtime_config = load_app_config()
    try:
        gpus = (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
        cpu_info = f"{platform.processor()} ({psutil.cpu_count(logical=True)} Cores)"
        ram_total_gb = psutil.virtual_memory().total / (1024**3)
        ram_available_gb = psutil.virtual_memory().available / (1024**3)
        ram_info = f"Total: {ram_total_gb:.2f} GB, Available: {ram_available_gb:.2f} GB"
        return jsonify(
            {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpus": gpus,
                "os": f"{platform.system()} {platform.release()}",
                "cpu": cpu_info,
                "ram": ram_info,
                "app_version": APP_VERSION,
                "default_output_folder": app_runtime_config.get("output_folder"),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/restart_backend", methods=["POST"])
def restart_backend_route():
    return jsonify({"message": "Backend restart initiated (placeholder)."}), 202


@app.route("/get_output_directory", methods=["GET"])
def get_output_directory_route():
    global app_runtime_config
    app_runtime_config = load_app_config()
    return jsonify({"output_directory": app_runtime_config.get("output_folder")})


@app.route("/set_output_directory", methods=["POST"])
def set_output_directory_route():
    global app_runtime_config
    data = request.get_json()
    new_path_str = data.get("output_directory")
    if not new_path_str:
        return jsonify({"error": "No directory path provided"}), 400
    try:
        new_path_resolved = Path(new_path_str).resolve()
        os.makedirs(new_path_resolved, exist_ok=True)
    except Exception as e:
        return jsonify({"error": f"Invalid directory path: {e}"}), 400
    if not os.access(str(new_path_resolved), os.W_OK):
        return jsonify({"error": "Directory path not writable."}), 400
    if save_app_config({"output_folder": str(new_path_resolved)}):
        app_runtime_config = load_app_config()
        app.config["OUTPUT_FOLDER"] = app_runtime_config["output_folder"]
        os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
        return jsonify(
            {
                "message": "Output directory updated.",
                "new_path": app_runtime_config["output_folder"],
            }
        )
    return jsonify({"error": "Failed to save configuration."}), 500


@app.route("/clear_backend_dirs", methods=["POST"])
def clear_backend_dirs_route():
    global app_runtime_config
    cleared_input_count = 0
    cleared_output_tasks_count = 0
    errors = []
    upload_folder = Path(app.config["UPLOAD_FOLDER"])
    try:
        for item in upload_folder.iterdir():
            if item.is_file():
                item.unlink()
                cleared_input_count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                cleared_input_count += 1
    except Exception as e:
        errors.append(f"Error clearing input '{upload_folder}': {e}")
    output_folder_root = Path(app_runtime_config["output_folder"])
    try:
        for item in output_folder_root.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                cleared_output_tasks_count += 1
            elif item.is_file():
                item.unlink()
    except Exception as e:
        errors.append(f"Error clearing output '{output_folder_root}': {e}")
    msg = f"Input: {cleared_input_count} items cleared. Output: {cleared_output_tasks_count} task folders cleared."
    return jsonify(
        {
            "message": (
                f"Clear successful. {msg}"
                if not errors
                else f"Clear partially successful. {msg}"
            ),
            "errors": errors,
        }
    ), (200 if not errors else 207)


if __name__ == "__main__":
    print(f"ScaleX WebUI v{APP_VERSION} starting. Access at http://127.0.0.1:5000")
    print(
        f"Input images will be temporarily stored in: {Path(app.config['UPLOAD_FOLDER']).resolve()}"
    )
    print(
        f"Output images will be saved under: {Path(app.config['OUTPUT_FOLDER']).resolve()}"
    )
    app.run(debug=False, host="127.0.0.1", port=5000)
