import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from config import OUTPUT_DIR

class MyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, 
                 calibration_data, 
                 batch_size=1, 
                 cache_file=OUTPUT_DIR  + "calibration.cache"):
        super().__init__()
        
        # Initialize CUDA context explicitly
        cuda.init()
        self.cuda_ctx = cuda.Device(0).make_context()  # Use first GPU
        
        self.batch_size = batch_size
        self.data = calibration_data
        self.current_index = 0
        self.cache_file = cache_file
        
        # Allocate memory with active context
        self.device_input = cuda.mem_alloc(self.data[0].nbytes)
        
        # Push context back
        self.cuda_ctx.pop()
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index >= len(self.data):
            return None
        
        # Push context
        self.cuda_ctx.push()
        
        batch = self.data[self.current_index]
        batch = np.ascontiguousarray(batch)
        
        cuda.memcpy_htod(self.device_input, batch)
        
        self.current_index += self.batch_size
        
        # Pop context
        self.cuda_ctx.pop()
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            
    def __del__(self):
        # Clean up context when object is destroyed
        try:
            self.cuda_ctx.detach()
        except:
            pass