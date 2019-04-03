# must sync with attribute.h and ctype_util.h

from ctypes import *

# typedef

BOOLEAN = c_char

# _MINI_TASK_NODE

class MINI_TASK_NODE(Structure):
    _fields_ = [("vm_id",          c_int),
                ("rt_type",        c_uint, 8),
                ("data_ptr",       c_ulonglong),

                ("node_id",        c_longlong),
                ("IsSwap",         BOOLEAN),
                ("IsHighPriority", BOOLEAN)
               ]

STOP_HANDLER = -100  # assign to vm_id
