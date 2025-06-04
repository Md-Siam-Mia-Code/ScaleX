# scalex/utils.py
import cv2
import os
import torch
import numpy as np
from torch import nn, Tensor
from typing import (
    Literal,
    Tuple,
    List,
    Optional,
    Union,
    OrderedDict,
    Callable,  # Import Callable for type hinting
    Dict,  # Import Dict for type hinting
    Any,  # Import Any for type hinting
)

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from .archs.gfpgan_bilinear_arch import GFPGANBilinear
from .archs.gfpganv1_arch import GFPGANv1
from .archs.gfpganv1_clean_arch import GFPGANv1Clean

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ArchType = Literal["clean", "original", "bilinear", "RestoreFormer"]


class ScaleXEnhancer:
    def __init__(
        self,
        model_path: str,
        upscale: float = 2.0,
        arch: ArchType = "clean",
        channel_multiplier: int = 2,
        bg_upsampler: Optional[nn.Module] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.upscale: float = upscale
        self.bg_upsampler: Optional[nn.Module] = bg_upsampler

        if device is None:
            self.device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.gfpgan: nn.Module
        # ... (rest of your __init__ method for model loading - NO CHANGES NEEDED HERE) ...
        if arch == "clean":
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "bilinear":
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "original":
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "RestoreFormer":
            try:
                from .archs.restoreformer_arch import RestoreFormer

                self.gfpgan = RestoreFormer()
            except ImportError:
                raise ImportError("RestoreFormer arch selected but module not found.")
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        facexlib_model_root_default = os.path.join(
            PROJECT_ROOT_DIR, "models", "weights"
        )
        if not os.path.isdir(facexlib_model_root_default):
            print(
                f"ScaleX Info: Pre-defined facexlib model rootpath '{facexlib_model_root_default}' not found."
            )
            facexlib_model_path_for_helper = facexlib_model_root_default
        else:
            facexlib_model_path_for_helper = facexlib_model_root_default

        self.face_helper = FaceRestoreHelper(
            upscale_factor=int(upscale),
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            model_rootpath=facexlib_model_path_for_helper,
        )

        if model_path.startswith("https://"):
            model_download_dir = os.path.join(PROJECT_ROOT_DIR, "models", "pretrained")
            os.makedirs(model_download_dir, exist_ok=True)
            model_name_from_url = model_path.split("/")[-1]
            local_model_path_check = os.path.join(
                model_download_dir, model_name_from_url
            )
            if os.path.isfile(local_model_path_check):
                print(
                    f"ScaleX Info: Model '{model_name_from_url}' found locally. Using local copy."
                )
                model_path = local_model_path_check
            else:
                print(f"ScaleX Info: Downloading model from URL: {model_path}")
                model_path = load_file_from_url(
                    url=model_path,
                    model_dir=model_download_dir,
                    progress=True,
                    file_name=None,
                )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        loadnet = torch.load(
            model_path, map_location=lambda storage, loc: storage, weights_only=True
        )
        keyname = (
            "params_ema"
            if "params_ema" in loadnet
            else (
                "params"
                if "params" in loadnet
                else ("g_ema" if "g_ema" in loadnet else None)
            )
        )
        loaded_successfully = False
        if keyname and keyname in loadnet:
            try:
                self.gfpgan.load_state_dict(loadnet[keyname], strict=False)
                loaded_successfully = True
            except RuntimeError as e:
                print(
                    f"ScaleX Warning: Failed to load state_dict with key '{keyname}'. Error: {e}"
                )
        if not loaded_successfully and isinstance(loadnet, (OrderedDict, dict)):
            try:
                new_state_dict = OrderedDict()
                has_module_prefix = any(k.startswith("module.") for k in loadnet.keys())
                if has_module_prefix:
                    print(
                        "ScaleX Info: Removing 'module.' prefix from model state_dict keys."
                    )
                    for k, v in loadnet.items():
                        name = k[7:] if k.startswith("module.") else k
                        new_state_dict[name] = v
                    self.gfpgan.load_state_dict(new_state_dict, strict=False)
                else:
                    self.gfpgan.load_state_dict(loadnet, strict=False)
                loaded_successfully = True
            except RuntimeError as e:
                print(f"ScaleX Warning: Failed to load state_dict directly. Error: {e}")
        if not loaded_successfully:
            raise ValueError(
                f"Cannot load model weights from {model_path}: Unknown format or key mismatch."
            )

        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)
        # End of __init__

    @torch.no_grad()
    def enhance(
        self,
        img: np.ndarray,
        has_aligned: bool = False,
        only_center_face: bool = False,
        paste_back: bool = True,
        weight: Optional[float] = None,  # Keep weight for GFPGANv1
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,  # ADDED
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:

        self.face_helper.clean_all()  # Reset face_helper for each image

        # --- STAGE: Face Detection & Alignment (if not has_aligned) ---
        if progress_callback:
            progress_callback({"event_type": "face_detection_start"})

        if has_aligned:
            if img.shape[0:2] != (512, 512):
                # print("ScaleX Info: Resizing aligned input...") # Optional log
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.cropped_faces = [img]
            # For aligned, number of faces is 1 (the input itself)
            if progress_callback:
                progress_callback(
                    {
                        "event_type": "face_detection_done",
                        "num_faces": 1,
                        "aligned_input": True,
                    }
                )
        else:
            self.face_helper.read_image(img)
            # The get_face_landmarks_5 method populates self.face_helper.all_landmarks_5
            self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, eye_dist_threshold=5
            )
            num_faces_detected = len(self.face_helper.all_landmarks_5)
            if progress_callback:
                progress_callback(
                    {
                        "event_type": "face_detection_done",
                        "num_faces": num_faces_detected,
                        "aligned_input": False,
                    }
                )

            if num_faces_detected > 0:
                if progress_callback:
                    progress_callback({"event_type": "face_alignment_start"})
                self.face_helper.align_warp_face()  # This populates self.face_helper.cropped_faces
                if progress_callback:
                    progress_callback(
                        {
                            "event_type": "face_alignment_done",
                            "num_aligned": len(self.face_helper.cropped_faces),
                        }
                    )

        # --- Handle cases where no faces are to be processed ---
        if not self.face_helper.cropped_faces:
            # print("ScaleX Info: No faces detected/aligned.") # Optional log
            if progress_callback:
                progress_callback({"event_type": "no_faces_to_process"})

            if paste_back:  # Try to return original or BG upsampled image
                if self.bg_upsampler is not None:
                    if progress_callback:
                        progress_callback({"event_type": "bg_upsample_only_start"})
                    try:
                        # print("ScaleX Info: No faces, attempting background upsampling only.") # Optional log
                        # Assuming bg_upsampler.enhance returns a tuple (upsampled_img, None) or just upsampled_img
                        bg_output_tuple_or_img = self.bg_upsampler.enhance(
                            img, outscale=self.upscale
                        )

                        bg_output_img = None
                        if (
                            isinstance(bg_output_tuple_or_img, tuple)
                            and len(bg_output_tuple_or_img) > 0
                            and isinstance(bg_output_tuple_or_img[0], np.ndarray)
                        ):
                            bg_output_img = bg_output_tuple_or_img[0]
                        elif isinstance(bg_output_tuple_or_img, np.ndarray):
                            bg_output_img = bg_output_tuple_or_img

                        if progress_callback:
                            progress_callback(
                                {
                                    "event_type": "bg_upsample_only_done",
                                    "success": bg_output_img is not None,
                                }
                            )

                        if bg_output_img is not None:
                            return [], [], bg_output_img
                        else:  # BG upsampling failed or returned unexpected type
                            return [], [], img.copy()  # Fallback to original
                    except Exception:  # Catch errors during BG upsample
                        if progress_callback:
                            progress_callback(
                                {
                                    "event_type": "bg_upsample_only_done",
                                    "success": False,
                                }
                            )
                        return [], [], img.copy()  # Fallback to original
                else:  # No BG upsampler, return original
                    return [], [], img.copy()
            else:  # Not pasting back and no faces
                return [], [], None

        # --- STAGE: Face Restoration Loop ---
        total_faces_to_process = len(self.face_helper.cropped_faces)
        for i, cropped_face_np in enumerate(self.face_helper.cropped_faces):
            if progress_callback:
                progress_callback(
                    {
                        "event_type": "processing_face_start",
                        "current_face": i + 1,
                        "total_faces": total_faces_to_process,
                    }
                )

            cropped_face_tensor: Tensor = img2tensor(
                cropped_face_np / 255.0, bgr2rgb=True, float32=True
            )
            normalize(
                cropped_face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
            )
            cropped_face_tensor = cropped_face_tensor.unsqueeze(0).to(self.device)

            gfpgan_call_kwargs = {"return_rgb": False}
            # Specific weight handling for GFPGANv1 based on your existing logic
            if weight is not None and isinstance(self.gfpgan, GFPGANv1):
                gfpgan_call_kwargs["weight"] = weight
            # elif weight is not None: # Optional: log if weight is ignored for other archs
            # print(f"ScaleX Warning: 'weight' ignored for arch {type(self.gfpgan).__name__}.")

            try:
                output_from_gfpgan = self.gfpgan(
                    cropped_face_tensor, **gfpgan_call_kwargs
                )
                # Handle different possible output structures from gfpgan
                if (
                    isinstance(output_from_gfpgan, tuple)
                    and len(output_from_gfpgan) > 0
                ):
                    output_tensor = output_from_gfpgan[0]
                elif isinstance(output_from_gfpgan, torch.Tensor):
                    output_tensor = output_from_gfpgan
                else:
                    raise ValueError(
                        f"Unexpected output type from GFPGAN: {type(output_from_gfpgan)}"
                    )

                restored_face_np = tensor2img(
                    output_tensor.squeeze(0), rgb2bgr=True, min_max=(-1, 1)
                )
            except Exception as e:
                # print(f"    [Error] ScaleX inference failed for face {i+1}: {e}") # Optional log
                restored_face_np = (
                    cropped_face_np  # Fallback to original cropped face on error
                )

            restored_face_np = restored_face_np.astype(np.uint8)
            self.face_helper.add_restored_face(
                restored_face_np
            )  # This appends to self.face_helper.restored_faces

            if progress_callback:
                progress_callback(
                    {
                        "event_type": "processing_face_done",
                        "current_face": i + 1,
                        "total_faces": total_faces_to_process,
                    }
                )

        # --- STAGE: Pasting Faces Back ---
        final_restored_img: Optional[np.ndarray] = None
        if not has_aligned and paste_back:
            if progress_callback:
                progress_callback({"event_type": "pasting_faces_start"})

            background_img_for_paste: Optional[np.ndarray] = None
            if self.bg_upsampler is not None:
                if progress_callback:
                    progress_callback({"event_type": "bg_upsample_for_paste_start"})
                try:
                    # print("ScaleX Info: Upsampling background for face pasting.") # Optional log
                    bg_output_tuple_or_img = self.bg_upsampler.enhance(
                        img, outscale=self.upscale
                    )

                    if (
                        isinstance(bg_output_tuple_or_img, tuple)
                        and len(bg_output_tuple_or_img) > 0
                        and isinstance(bg_output_tuple_or_img[0], np.ndarray)
                    ):
                        background_img_for_paste = bg_output_tuple_or_img[0]
                    elif isinstance(bg_output_tuple_or_img, np.ndarray):
                        background_img_for_paste = bg_output_tuple_or_img

                    if progress_callback:
                        progress_callback(
                            {
                                "event_type": "bg_upsample_for_paste_done",
                                "success": background_img_for_paste is not None,
                            }
                        )
                except Exception:  # Catch errors during BG upsample
                    if progress_callback:
                        progress_callback(
                            {
                                "event_type": "bg_upsample_for_paste_done",
                                "success": False,
                            }
                        )

            # Ensure restored_faces is populated before pasting
            if not self.face_helper.restored_faces and self.face_helper.cropped_faces:
                # print("ScaleX Warning: Faces cropped, but none restored. Pasting may use original faces.") # Optional log
                pass  # FaceRestoreHelper might handle this by pasting original cropped if restored is empty.

            self.face_helper.get_inverse_affine(None)  # Prepare for pasting
            final_restored_img = self.face_helper.paste_faces_to_input_image(
                upsample_img=background_img_for_paste  # Pass the (potentially None) upsampled BG
            )
            if progress_callback:
                progress_callback({"event_type": "pasting_faces_done"})

        elif has_aligned and self.face_helper.restored_faces:
            # If input was_aligned, the first (and only) restored face is the result
            final_restored_img = self.face_helper.restored_faces[0]
            if progress_callback:
                progress_callback({"event_type": "final_output_ready_aligned"})

        elif not paste_back and self.face_helper.restored_faces:
            # Not pasting back, but faces were processed.
            # If only one face was processed (e.g. from only_center_face or a single detected face),
            # then that single restored face could be considered the 'final_restored_img'.
            # Otherwise, 'final_restored_img' remains None, and caller uses the list of restored_faces.
            if len(self.face_helper.restored_faces) == 1:
                final_restored_img = self.face_helper.restored_faces[0]
            if progress_callback:
                progress_callback({"event_type": "final_outputs_ready_no_paste"})

        # --- Return all processed data ---
        return (
            self.face_helper.cropped_faces,
            self.face_helper.restored_faces,
            final_restored_img,
        )
