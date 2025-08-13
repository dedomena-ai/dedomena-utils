import json
import numpy as np
from fastapi.encoders import jsonable_encoder

def json_serialize(data):
    """
    Converts any complex object into a JSON serializable format and returns a JSON string.
    
    - Converts NumPy types to native Python types.
    - Applies FastAPI's jsonable_encoder for BaseModel, datetime, lists, dicts, etc.
    - Returns a JSON string.
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
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    # First, convert NumPy types to native Python types
    safe_data = convert_numpy(data)
    # Then apply jsonable_encoder
    json_ready = jsonable_encoder(safe_data)
    return json_ready
