import ctypes
from ctypes import c_void_p, c_size_t, c_int, Structure, CFUNCTYPE
from .llaisys_types import *

# Define function pointer types
get_device_count_api = CFUNCTYPE(c_int)
set_device_api = CFUNCTYPE(None, c_int)
device_synchronize_api = CFUNCTYPE(None)

create_stream_api = CFUNCTYPE(llaisysStream_t)
destroy_stream_api = CFUNCTYPE(None, llaisysStream_t)
stream_synchronize_api = CFUNCTYPE(None, llaisysStream_t)
create_event_api = CFUNCTYPE(llaisysEvent_t)
destroy_event_api = CFUNCTYPE(None, llaisysEvent_t)
event_record_api = CFUNCTYPE(None, llaisysEvent_t, llaisysStream_t)
stream_wait_event_api = CFUNCTYPE(None, llaisysStream_t, llaisysEvent_t)
event_synchronize_api = CFUNCTYPE(None, llaisysEvent_t)

malloc_device_api = CFUNCTYPE(c_void_p, c_size_t)
free_device_api = CFUNCTYPE(None, c_void_p)
malloc_host_api = CFUNCTYPE(c_void_p, c_size_t)
free_host_api = CFUNCTYPE(None, c_void_p)

memcpy_sync_api = CFUNCTYPE(None, c_void_p, c_void_p, c_size_t, llaisysMemcpyKind_t)
memcpy_async_api = CFUNCTYPE(None, c_void_p, c_void_p, c_size_t, llaisysMemcpyKind_t, llaisysStream_t)


# Define the struct matching LlaisysRuntimeAPI
class LlaisysRuntimeAPI(Structure):
    _fields_ = [
        ("get_device_count", get_device_count_api),
        ("set_device", set_device_api),
        ("device_synchronize", device_synchronize_api),
        ("create_stream", create_stream_api),
        ("destroy_stream", destroy_stream_api),
        ("stream_synchronize", stream_synchronize_api),
        ("create_event", create_event_api),
        ("destroy_event", destroy_event_api),
        ("event_record", event_record_api),
        ("stream_wait_event", stream_wait_event_api),
        ("event_synchronize", event_synchronize_api),
        ("malloc_device", malloc_device_api),
        ("free_device", free_device_api),
        ("malloc_host", malloc_host_api),
        ("free_host", free_host_api),
        ("memcpy_sync", memcpy_sync_api),
        ("memcpy_async", memcpy_async_api),
    ]


# Load shared library
def load_runtime(lib):
    # Declare API function prototypes
    lib.llaisysGetRuntimeAPI.argtypes = [llaisysDeviceType_t]
    lib.llaisysGetRuntimeAPI.restype = ctypes.POINTER(LlaisysRuntimeAPI)

    lib.llaisysSetContextRuntime.argtypes = [llaisysDeviceType_t, c_int]
    lib.llaisysSetContextRuntime.restype = None
