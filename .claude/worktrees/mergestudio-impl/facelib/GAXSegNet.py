from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from core.interact import interact as io
from core.leras import nn


class GAXSegNet(object):
    """PyTorch-only GA-XSeg wrapper with DFL-compatible API.
    Same interface as XSegNet for drop-in compatibility.
    """

    VERSION = 1

    def __init__(
        self,
        name,
        resolution=256,
        use_eca=True,
        load_weights=True,
        weights_file_root=None,
        training=False,
        place_model_on_cpu=False,
        run_on_cpu=False,
        optimizer=None,
        data_format="NCHW",
        raise_on_no_model_files=False,
    ):

        self.resolution = int(resolution)
        self.weights_file_root = Path(weights_file_root) if weights_file_root is not None else Path(__file__).parent
        self.training = bool(training)
        self._save_prefix = name  # model-name prefix for consistent save/backup paths

        nn.initialize_main_env()
        nn.set_data_format("NCHW")

        torch = nn.torch

        res_str = str(self.resolution)
        self.model_filename_list = []

        if run_on_cpu or place_model_on_cpu:
            model_device = torch.device("cpu")
        else:
            model_device = nn.device

        prev_nn_device = nn.device
        try:
            nn.device = model_device
            self.model = nn.GAXSeg(3, 32, 1, use_eca=use_eca, name=name)
            self.model_weights = self.model.get_weights()

            if self.training:
                if optimizer is None:
                    raise ValueError("Optimizer should be provided for training mode.")
                self.opt = optimizer
                self.opt.initialize_variables(self.model_weights, vars_on_cpu=bool(place_model_on_cpu))
                self.model_filename_list += [[self.opt, f"{res_str}_opt.pth"]]
        finally:
            nn.device = prev_nn_device

        self.model_filename_list += [[self.model, f"{res_str}.pth"]]

        if not self.training:
            self.model.eval()

            def net_run(input_np: np.ndarray) -> np.ndarray:
                x = self._np_to_torch_nchw(input_np, device=model_device)
                with torch.no_grad():
                    _, pred = self.model(x)
                out = pred.detach().to("cpu").numpy()
                return out

            self.net_run = net_run

        self.initialized = True
        for mdl, filename in self.model_filename_list:
            do_init = not load_weights

            if not do_init:
                model_file_path = self.weights_file_root / (self._save_prefix + '_' + filename)

                if mdl is self.model:
                    ok = self._load_model_weights_compat(model_file_path)
                else:
                    ok = mdl.load_weights(model_file_path)

                do_init = not ok
                if do_init:
                    if raise_on_no_model_files:
                        raise Exception(f"{model_file_path} does not exist or failed to load.")
                    if not self.training:
                        self.initialized = False
                        break

            if do_init:
                mdl.init_weights()

    def get_resolution(self):
        return self.resolution

    def flow(self, x, pretrain=False):
        return self.model(x, pretrain=pretrain)

    def get_weights(self):
        return self.model_weights

    def save_weights(self):
        for mdl, filename in io.progress_bar_generator(self.model_filename_list, "Saving", leave=False):
            save_path = self.weights_file_root / (self._save_prefix + '_' + filename)
            mdl.save_weights(save_path)

    def extract(self, input_image: np.ndarray):
        if not self.initialized:
            return 0.5 * np.ones((self.resolution, self.resolution, 1), dtype=np.float32)

        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[None, ...]

        result = np.clip(self.net_run(input_image), 0.0, 1.0)
        result[result < 0.1] = 0.0

        result = np.transpose(result, (0, 2, 3, 1))

        if input_shape_len == 3:
            result = result[0]

        return result

    def _np_to_torch_nchw(self, arr: np.ndarray, device):
        torch = nn.torch
        if arr.ndim == 3:
            arr = arr[None, ...]

        if arr.ndim != 4:
            raise ValueError(f"expected 3D/4D image array, got shape {arr.shape}")

        if arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))

        x = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        return x

    def _load_model_weights_compat(self, filename: Path) -> bool:
        """Load weights with basic TF->Torch layout conversions."""
        filename = Path(filename)
        if not filename.exists():
            alt = None
            if filename.suffix == '.pth':
                alt = filename.with_suffix('.npy')
            elif filename.suffix == '.npy':
                alt = filename.with_suffix('.pth')
            if alt is not None and alt.exists():
                filename = alt
            else:
                return False

        try:
            d = pickle.loads(filename.read_bytes())
        except Exception:
            return False

        torch = nn.torch
        params = self.model.get_weights()

        with torch.no_grad():
            for i, p in enumerate(params):
                key = f"param_{i}"
                w_val = d.get(key, None)
                if w_val is None:
                    continue

                if not isinstance(w_val, np.ndarray):
                    w_val = np.array(w_val)

                target_shape = tuple(p.shape)
                src = self._convert_weight_to_shape(w_val, target_shape)
                if src is None:
                    if int(np.prod(w_val.shape)) == int(np.prod(target_shape)):
                        src = w_val.reshape(target_shape)
                    else:
                        return False

                t = torch.from_numpy(src).to(device=p.device, dtype=p.dtype)
                p.data.copy_(t)

        return True

    def _convert_weight_to_shape(self, w: np.ndarray, target_shape: tuple) -> Optional[np.ndarray]:
        if tuple(w.shape) == tuple(target_shape):
            return w

        if w.ndim == 1:
            if int(np.prod(w.shape)) == int(np.prod(target_shape)):
                return w.reshape(target_shape)
            return None

        if w.ndim == 2:
            if w.T.shape == tuple(target_shape):
                return w.T
            if int(np.prod(w.shape)) == int(np.prod(target_shape)):
                return w.reshape(target_shape)
            return None

        if w.ndim == 4:
            candidates = [
                w,
                np.transpose(w, (3, 2, 0, 1)),
                np.transpose(w, (2, 3, 0, 1)),
                np.transpose(w, (1, 0, 2, 3)),
                np.transpose(w, (0, 1, 3, 2)),
            ]
            for c in candidates:
                if tuple(c.shape) == tuple(target_shape):
                    return c
            if int(np.prod(w.shape)) == int(np.prod(target_shape)):
                return w.reshape(target_shape)
            return None

        if int(np.prod(w.shape)) == int(np.prod(target_shape)):
            return w.reshape(target_shape)
        return None
