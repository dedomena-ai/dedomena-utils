import json
import numpy as np
from fastapi.encoders import jsonable_encoder

def json_serialize(data):
    """
    Converts any complex object into a JSON-serializable format.
    
    - Converts NumPy types to native Python types.
    - Replaces NaN, Inf, -Inf with None (JSON valid).
    - Applies FastAPI's jsonable_encoder.
    """

    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return convert_numpy(obj.tolist())
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            # Manejar NaN e Inf
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, float):
            # Por si vienen floats normales de Python
            if obj in (float("inf"), float("-inf")) or obj != obj:  # obj != obj -> NaN
                return None
            return obj
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    safe_data = convert_numpy(data)
    json_ready = jsonable_encoder(safe_data)
    return json_ready