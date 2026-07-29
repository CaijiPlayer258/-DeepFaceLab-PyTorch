import pickle
from pathlib import Path
from core import pathex
import numpy as np
import torch
import os

from core.leras.nn import nn

class Saveable():
    def __init__(self, name=None):
        self.name = name

    #override
    def get_weights(self):
        #return torch parameters that should be initialized/loaded/saved
        return []

    #override
    def get_weights_np(self):
        weights = self.get_weights()
        if len(weights) == 0:
            return []
        return [w.detach().cpu().to(torch.float32).numpy() if w.dtype == torch.bfloat16 else w.detach().cpu().numpy() for w in weights]

    def set_weights(self, new_weights):
        weights = self.get_weights()
        if len(weights) != len(new_weights):
            raise ValueError ('len of lists mismatch')

        for w, new_w in zip(weights, new_weights):
            if isinstance(new_w, torch.nn.Parameter) or isinstance(new_w, torch.Tensor):
                src = new_w.data if hasattr(new_w, 'data') else new_w
                src = src.to(device=w.device, dtype=w.dtype)
                w.data.copy_(src)
            else:
                if not isinstance(new_w, np.ndarray):
                    new_w = np.array(new_w)
                src = torch.from_numpy(new_w).reshape(w.shape).to(device=w.device, dtype=w.dtype)
                w.data.copy_(src)

    def save_weights(self, filename, force_dtype=None):
        d = {}
        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        name = self.name

        for i, w in enumerate(weights):
            w_t = w.detach().cpu()
            if w_t.dtype == torch.bfloat16:
                w_t = w_t.to(torch.float32)
            w_val = w_t.numpy().copy()
            
            if force_dtype is not None:
                w_val = w_val.astype(force_dtype)

            w_name = f"param_{i}"
            d[w_name] = w_val

        # Stream pickle directly to disk to avoid an extra in-memory copy
        # of the entire weights dict (pickle.dumps), which can trigger
        # MemoryError on large models.
        p = Path(filename)
        p_tmp = p.parent / (p.name + '.tmp')
        with open(p_tmp, 'wb') as f:
            pickle.dump(d, f, protocol=4)
        if p.exists():
            p.unlink()
        p_tmp.rename(p)

    def load_weights(self, filename):
        """
        returns True if file exists
        """
        filepath = Path(filename)

        if not filepath.exists():
            alt = None
            if filepath.suffix == '.pth':
                alt = filepath.with_suffix('.npy')
            elif filepath.suffix == '.npy':
                alt = filepath.with_suffix('.pth')
            if alt is not None and alt.exists():
                filepath = alt

        if not filepath.exists():
            return False

        try:
            with open(filepath, 'rb') as f:
                d = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError, OSError) as e:
            try:
                with open(filepath, 'rb') as f:
                    head = f.read(32)
            except Exception:
                head = b''
            print(
                f"[WARN] Failed to load weights from '{os.fspath(filepath)}': {e}. "
                f"This file does not look like a pickle weights file. "
                f"Header bytes: {head!r}. "
                f"You may need to delete/replace the corrupted or wrong-format weights file."
            )
            return False

        # Detect original DFL (TensorFlow) format: keys contain TF scope names
        if any(isinstance(k, str) and ':0' in k for k in d.keys()):
            return self._load_tf_weights(d)

        return self._load_pt_weights(d)

    def _load_pt_weights(self, d):
        """Load current-format weights (param_0, param_1, ...)."""
        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        try:
            for i, w in enumerate(weights):
                w_name = f"param_{i}"
                w_val = d.get(w_name, None)

                if w_val is None:
                    pass
                else:
                    w_val = np.reshape(w_val, w.shape)
                    src = torch.from_numpy(w_val).to(device=w.device, dtype=w.dtype)
                    w.data.copy_(src)
        except:
            return False

        return True

    def _load_tf_weights(self, d):
        """Load original DeepFaceLab (TensorFlow) .npy weights.

        Mirrors tools/convert_dfl_tf_to_torch.py logic:
          1. Build TF name → param mapping from layer hierarchy
          2. Try exact name match first
          3. Fall back to shape-based greedy matching
          4. Two-pass: verify ALL params match before applying any
        """
        if not hasattr(self, '_get_tf_weight_names'):
            return False
        tf_weights = self._get_tf_weight_names()
        if not tf_weights:
            return False

        remaining_keys = set(d.keys())
        validated: List[object] = []

        def _convert_to_shape(src: np.ndarray, dst_shape) -> np.ndarray:
            dst_shape = tuple(int(x) for x in dst_shape)
            if tuple(src.shape) == dst_shape:
                return src
            # Conv2D TF NHWC (H,W,in,out) -> Torch NCHW (out,in,H,W)
            if src.ndim == 4 and len(dst_shape) == 4:
                if (src.shape[0], src.shape[1], src.shape[2], src.shape[3]) == (
                    dst_shape[2], dst_shape[3], dst_shape[1], dst_shape[0]):
                    return np.transpose(src, (3, 2, 0, 1))
            # Dense transpose fallback
            if src.ndim == 2 and len(dst_shape) == 2:
                if src.shape[::-1] == dst_shape:
                    return src.T
            # Resize if element count matches
            if int(np.prod(src.shape)) == int(np.prod(dst_shape)):
                return np.reshape(src, dst_shape)
            raise ValueError(f"shape mismatch: src {src.shape} -> dst {dst_shape}")

        def _try_key(key: str, dst_shape) -> object:
            if key not in remaining_keys:
                return None
            try:
                return _convert_to_shape(d[key], dst_shape)
            except ValueError:
                return None

        # --- Pass 1: name-based + shape-fallback matching ---
        for tf_name, param in tf_weights:
            dst_shape = tuple(int(x) for x in param.shape)
            chosen = None

            # 1a) exact name
            chosen = _try_key(tf_name, dst_shape)

            # 1b) without / with :0 suffix
            if chosen is None:
                if tf_name.endswith(':0'):
                    chosen = _try_key(tf_name[:-2], dst_shape)
                else:
                    chosen = _try_key(tf_name + ':0', dst_shape)

            # 1c) shape-based greedy fallback among unused keys
            if chosen is None:
                for cand in list(remaining_keys):
                    chosen = _try_key(cand, dst_shape)
                    if chosen is not None:
                        remaining_keys.remove(cand)
                        break

            if chosen is None:
                name_hint = self.name or type(self).__name__
                print(f"[WARN] TF weight '{tf_name}' (shape {param.shape}) 在 {name_hint} 中无匹配，跳过加载")
                return False  # refuse partial load

            validated.append((param, chosen))

        # --- Pass 2: apply ---
        for param, w_val in validated:
            w_val = np.reshape(w_val, param.shape)
            param.data.copy_(
                torch.from_numpy(w_val).to(device=param.device, dtype=param.dtype)
            )

        return True

    def init_weights(self):
        # PyTorch initializes weights automatically
        pass

nn.Saveable = Saveable
