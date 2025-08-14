# src/utils.py

import torch
import structlog

logger = structlog.get_logger()

def get_safe_gpu_utilization(buffer_gb: float = 1.0) -> float:
    """
    Calculates a safe GPU memory utilization percentage for vLLM,
    with conservative estimates for systems with limited VRAM.

    Args:
        buffer_gb (float): The amount of memory in GB to leave as a buffer.

    Returns:
        float: A utilization percentage for vLLM, or 0.0 for CPU fallback.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Returning 0.0 for CPU fallback.")
        return 0.0

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    torch.cuda.empty_cache()
    
    free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(0)
    free_mem_gb = free_mem_bytes / (1024**3)
    used_mem_gb = (total_mem_bytes - free_mem_bytes) / (1024**3)
    
    logger.info(f"GPU Memory Status: Total={total_mem_gb:.2f}GB, Used={used_mem_gb:.2f}GB, Free={free_mem_gb:.2f}GB")

    if total_mem_gb <= 4.0:
        logger.warning(f"Low VRAM detected ({total_mem_gb:.1f}GB). Using CPU fallback for LLM.")
        return 0.0
    
    if used_mem_gb > 0.5:
        logger.warning(f"Significant GPU memory already in use ({used_mem_gb:.2f}GB). Using CPU fallback for LLM.")
        return 0.0

    safe_to_use_gb = free_mem_gb - buffer_gb
    
    if safe_to_use_gb <= 1.0:
        logger.warning(f"Insufficient free GPU memory for vLLM ({free_mem_gb:.2f}GB free). Using CPU fallback.")
        return 0.0

    utilization = safe_to_use_gb / total_mem_gb
    
    safe_utilization = max(0.1, min(utilization, 0.8))
    
    logger.info(f"Calculated safe GPU memory utilization for vLLM: {safe_utilization:.2f}")
    return safe_utilization

def cleanup_gpu_memory():
    """Clean up GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU memory cache cleared")

def get_recommended_model_for_memory(total_vram_gb: float) -> str:
    """
    Recommend an appropriate model based on available VRAM
    
    Args:
        total_vram_gb: Total VRAM in GB
    
    Returns:
        str: Recommended model name
    """
    if total_vram_gb <= 4:
        return "microsoft/phi-2"
    elif total_vram_gb <= 8:
        return "microsoft/Phi-3.5-mini-instruct"
    elif total_vram_gb <= 16:
        return "mistralai/Mistral-7B-Instruct-v0.3"
    else:
        return "meta-llama/Llama-2-13b-chat-hf"

def check_memory_requirements(model_name: str) -> dict:
    """
    Check if current system can handle the specified model
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        dict: Memory check results
    """
    if not torch.cuda.is_available():
        return {
            "can_run_gpu": False,
            "can_run_cpu": True,
            "recommendation": "CPU execution recommended",
            "total_vram": 0
        }
    
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_mem_bytes, _ = torch.cuda.mem_get_info(0)
    free_mem_gb = free_mem_bytes / (1024**3)
    
    model_sizes = {
        "microsoft/phi-2": 1.4,
        "microsoft/Phi-3.5-mini-instruct": 2.0,
        "mistralai/Mistral-7B-Instruct-v0.3": 3.5,
        "meta-llama/Llama-2-7b-chat-hf": 3.5,
        "meta-llama/Llama-2-13b-chat-hf": 6.5,
    }
    
    estimated_size = model_sizes.get(model_name, 2.0)
    
    can_run_gpu = free_mem_gb > (estimated_size + 0.5)
    
    recommended_model = get_recommended_model_for_memory(total_mem_gb)
    
    return {
        "can_run_gpu": can_run_gpu,
        "can_run_cpu": True,
        "estimated_model_size_gb": estimated_size,
        "free_vram_gb": free_mem_gb,
        "total_vram": total_mem_gb,
        "recommended_model": recommended_model,
        "recommendation": f"GPU: {'Yes' if can_run_gpu else 'No'}, CPU: Yes"
    }