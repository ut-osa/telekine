# must sync with attribute.h and ctype_util.h

import fcntl, types
import dill as pickle
#import cloudpickle as pickle

from devconf import *
from ctypes import *

# typedef

BOOLEAN = c_char
HANDLE = c_uint64

PVOID = c_void_p

# _PARAM_BASE

class OPENCL_SWAP_HANDLE(Structure):
    _fields_ = [("Context",      HANDLE),
                ("MemoryFlag",   HANDLE),
                ("CommandQueue", HANDLE)
               ]

class SWAP_HANDLE(Union):
    _fields_ = [("OpenCLSwapHandle", OPENCL_SWAP_HANDLE)]

class PARAM_BASE(Structure):
    _fields_ = [("rt_type",   c_uint8),
                ("cmd_id",    c_uint8),
                ("FreeCmdId", c_uint8),
                ("dep",       c_int32),
                ("object_id", c_int32),

                # Flags
                ("done",       c_uint8),
                ("async",      c_uint8, 1),
                ("barrier",    c_uint8, 1),
                ("rate_limit", c_uint8, 2),
                ("mem_alloc",  c_uint8, 1),
                ("mem_free",   c_uint8, 1),

                # Size
                ("dstore_offset", c_uint64),
                ("dstore_size",   c_size_t),
                ("param_size",    c_size_t),

                # Callback
                ("guest_callback_param_addr", c_uint64),
                ("callback_param_offset",     c_int64),
                ("class_id",                  c_int32),

                # Memory allocation.
                ("OriginalObjectHandle", HANDLE),
                ("ObjectHandle",         HANDLE),
                ("SwappedOutAddress",    PVOID),
                ("AllocatedMemorySize",  c_size_t),

                # Runtime-dependent
                ("", SWAP_HANDLE)
               ]

class PARAM_PAYLOAD(Structure):
    _fields_ = [("addr",   POINTER(c_char)),
                ("size",   c_size_t),
                ("offset", c_uint64)
               ]

class TF_PY_PARAM(Structure):
    _fields_ = [("base", PARAM_BASE),

                # Payload
                ('param1', PARAM_PAYLOAD),
                ('param2', PARAM_PAYLOAD),
                ('param3', PARAM_PAYLOAD),
                ('param4', PARAM_PAYLOAD),
                ('param5', PARAM_PAYLOAD),
                ('param6', PARAM_PAYLOAD),
                ('param7', PARAM_PAYLOAD),
                ('param8', PARAM_PAYLOAD),
                ('param9', PARAM_PAYLOAD),
                ('param10', PARAM_PAYLOAD),
                ('param11', PARAM_PAYLOAD),
                ('param12', PARAM_PAYLOAD),

                ("ret_val1", PARAM_PAYLOAD)
               ]


# Variables

callback_dict = {}


# Utilities

def InitTFParam(cmd_id):
    param = TF_PY_PARAM()
    param.base.rt_type = TENSORFLOW_PY_API
    param.base.cmd_id = cmd_id
    param.base.dep = -1
    param.base.object_id = -1
    param.base.done = STATUS_TASK_UNDEFINED
    param.base.param_size = sizeof(TF_PY_PARAM)
    return param


def InitTFCallback():
    param = TF_PY_PARAM()
    param.base.cmd_id = TF_PY_NW_CALLBACK_INIT
    param.base.done = STATUS_TASK_UNDEFINED
    param.base.param_size = sizeof(TF_PY_PARAM)
    param.param1.size = 65536
    param.param2.size = 1024
    param.ret_val1.size = 1024
    param.base.dstore_size = 65536 + 1024 + 1024
    return param


def InitTFParamWithCallback(cmd_id):
    param = InitTFParam(cmd_id)
    callback_param = InitTFCallback()
    param.base.guest_callback_param_addr = addressof(callback_param)
    return param, callback_param


def SerializeParam(param, payload, obj, update=True):
    if obj is None:
        payload.size = 0
        return None, None

    dump = pickle.dumps(obj, recurse=True)
    payload.size = len(dump)
    if update:
        param.base.dstore_size += payload.size
    buf = create_string_buffer(dump, len(dump))
    payload.addr = pointer(c_char.from_address(addressof(buf)))
    return dump, buf


def ReserveReturnValue(param, payload):
    buf = create_string_buffer(1024) # HACK: reserve enough buffer
    payload.addr = pointer(c_char.from_address(addressof(buf)))
    payload.size = len(buf)
    param.base.dstore_size += payload.size
    return buf


def CreatePayloadBuffer(payload):
    print("create payload buffer of size %ld" % payload.size)
    buf = create_string_buffer(payload.size)
    payload.addr = pointer(c_char.from_address(addressof(buf)))
    return buf


def IsMethod(obj):
    return isinstance(obj, types.MethodType) and \
        obj.im_class is not NwObject


def IoctlWrapper(param, callback_param):
    ret = fcntl.ioctl(get_vgpu_fd(), IOCTL_TF_PY_CMD, param, 1)
    if ret == 0:
        return

    if ret < 0:
        print("ioctl TF_PY_cmd failed: %d\n" % ret);
        exit(-1);

    while (ret == STATUS_TASK_CALLBACK):
        print("receive a callback")

        # allocate buffers for callback parameters
        # assume callback method only takes *args and **kwargs
        args_buf = CreatePayloadBuffer(callback_param.param1)
        kwargs_buf = CreatePayloadBuffer(callback_param.param2)

        # notify guestdrv to fill out the buffers
        callback_param.base.done = STATUS_CALLBACK_POLL
        callback_param.base.cmd_id = TF_PY_NW_CALLBACK
        callback_ret = fcntl.ioctl(get_vgpu_fd(), IOCTL_TF_PY_CALLBACK,
                                   callback_param, 1)
        if callback_ret != STATUS_CALLBACK_FILLED:
            print("ioctl TF_PY_callback failed: %d\n" % callback_ret);
            exit(-1);

        # unpickle callback method and parameters
        args_buf = pickle.loads(args_buf)
        kwargs_buf = pickle.loads(kwargs_buf)
        callback_func = callback_dict[callback_param.base.object_id]
        print(callback_func, args_buf, kwargs_buf)

        # execute callback and pickle the result
        callback_ret_value = callback_func(*args_buf, **kwargs_buf)
        print("callback ret_val=", callback_ret_value)
        ret_buf, ret_dump = SerializeParam(callback_param,
                                           callback_param.ret_val1,
                                           callback_ret_value,
                                           update=False)

        # notify worker to receive the result
        callback_param.base.done = STATUS_CALLBACK_DONE
        callback_ret = fcntl.ioctl(get_vgpu_fd(), IOCTL_TF_PY_CALLBACK,
                                   callback_param, 1)
        if callback_ret != STATUS_CALLBACK_DONE:
            print("ioctl TF_PY_callback failed: %d\n" % callback_ret);
            exit(-1);

        # notify guestdrv to restore context
        param.base.done = STATUS_TASK_CONTINUE
        ret = fcntl.ioctl(get_vgpu_fd(), IOCTL_TF_PY_CMD, param, 1)
        if ret < 0:
            print("ioctl TF_PY_cmd failed: %d\n" % ret);
            exit(-1);


# Serialize class instance if obj is a method
'''
def SerializeMethod(method):
    param = InitTFParam(TF_PY_NW_METHOD)

    ins = method.im_self
    name = method.__name__
    print("method info", ins, name)
    dump1, buf1 = SerializeParam(param, param.param1, ins)
    dump2, buf2 = SerializeParam(param, param.param2, name)

    ret = fcntl.ioctl(get_vgpu_fd(), IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd serializeMethod failed: %d\n" % ret);
        exit(-1);

    print("method object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id, True)
'''


def SerializeMethod(method, cid = 0):
    param = InitTFParam(TF_PY_NW_METHOD)
    param.base.class_id = cid

    ret = fcntl.ioctl(get_vgpu_fd(), IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd serializeMethod failed: %d\n" % ret);
        exit(-1);

    print("method object_id=%d, class_id=%d" % (param.base.object_id, cid))
    callback_dict[param.base.object_id] = method
    return NwObject(param.base.object_id, class_id=cid, is_method=True)


def CallbackWhiteList(method, key):
    # TODO: also check class type
    # TODO: enable tf.data.Dataset.map and tf.contrib.data.parallel_interleave
    # to test nested callback
    if (method == "evaluate" and key == "input_fn") or \
       (method == "train" and key == "input_fn"):
        return True
    return False


# Local stub

class NwObject(object):
    def __init__(self, _id = 0, class_id = 0, is_method = False):
        self._object_id = _id
        self._class_id = class_id
        self._is_method = is_method

    def object_id(self):
        return self._object_id

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            return super(NwObject, self).__getattr__(name)

        def run(*args, **kwargs):
            param, cb_param = InitTFParamWithCallback(TF_PY_NW_OBJECT)
            param.base.object_id = self._object_id

            if kwargs is not None:
                for k, v in kwargs.items():
                    # TODO: not all function parameters are callbacks
                    if IsMethod(v) and CallbackWhiteList(name, k):
                        print("serialize methods", v)
                        kwargs[k] = SerializeMethod(v, self._class_id)

            dump1, buf1 = SerializeParam(param, param.param1, name)
            if args is not None:
                print(name, "pack args", args)
                dump2, buf2 = SerializeParam(param, param.param2, args)
            if kwargs is not None:
                print(name, "pack kwargs", kwargs)
                dump3, buf3 = SerializeParam(param, param.param3, kwargs)

            buf = ReserveReturnValue(param, param.ret_val1)

            IoctlWrapper(param, cb_param)

            if param.base.object_id > 0:
                return NwObject(param.base.object_id)
            if param.ret_val1.size > 0:
                return pickle.loads(buf)
            return None

        return run

    # TODO: read https://pythonhosted.org/Pyro4/
