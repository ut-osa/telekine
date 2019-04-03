import sys, os, fcntl, mmap, time
import traceback
import threading, Queue
from absl import flags

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import tpu
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

from structs import *
sys.path.insert(1, os.path.join(sys.path[0], '../include/common'))
from devconf import *
from tensorflow_py import *


initialized = False


def parse_param(vm_id, mm, param, payload):
    if payload.size == 0:
        return None

    print("parse_param: vm_id=%d, cmd=%d, obj=%d, dstore=%lx,offset=%x, payload_size=%d, payload_offset=%d" %
          (vm_id,
           param.base.cmd_id,
           param.base.object_id,
           param.base.dstore_size, param.base.dstore_offset,
           payload.size, payload.offset))

    mm.seek(INVOKER_FIFO_SIZE + VGPU_DSTORE_SIZE * (vm_id - 1) +
            param.base.dstore_offset + payload.offset)
    param_dump = mm.read(payload.size)
    print("load:", pickle.loads(param_dump))
    return pickle.loads(param_dump)


def writeback_result(vm_id, mm, param, payload, arg):
    print "writeback", arg, "at dstore_offset =", param.base.dstore_offset, \
          "offset =", payload.offset, "size =", payload.size
    dump = pickle.dumps(arg)
    payload.size = len(dump)
    print "writeback dump size =", payload.size
    mm.seek(INVOKER_FIFO_SIZE +
            VGPU_DSTORE_SIZE * (vm_id - 1) +
            param.base.dstore_offset +
            payload.offset)
    mm.write(dump)


def is_unpickleable_type(obj):
    if isinstance(obj,
            (tf.Session, tf.Tensor, tf.Operation, tf.data.Dataset,
             tf.SparseTensor, tf.data.Iterator, tf.contrib.tpu.TPUEstimator)):
        return True
    return False


def tuple_mapper(t, index_list):
    global object_dict
    global object_id

    if (t is None) or (not isinstance(t, tuple)):
        return t

    l = list(t)
    for index in index_list:
        obj = l[index]
        if is_unpickleable_type(obj):
            object_dict[object_id] = obj
            l[index] = NwObject(object_id)
            object_id += 1

    return tuple(l)


def dict_mapper(d):
    global object_dict
    global object_id

    if (d is None) or (type(d) is not dict):
        return
    for k, v in d.items():
        if isinstance(v, dict):
            dict_mapper(d[k])
        else:
            if is_unpickleable_type(v):
                object_dict[object_id] = v
                d[k] = NwObject(object_id)
                object_id += 1


def list_mapper(l):
    global object_dict
    global object_id

    if (l is None) or (not isinstance(l, list)):
        return
    for i in range(len(l)):
        if isinstance(l[i], list):
            list_mapper(l[i])
        else:
            if is_unpickleable_type(l[i]):
                object_dict[object_id] = l[i]
                l[i] = NwObject(object_id)
                object_id += 1


def tuple_walker(t):
    if (t is None) or (not isinstance(t, tuple)):
        return

    l = list(t)
    for i in range(len(l)):
        if isinstance(l[i], tuple):
            l[i] = tuple_walker(l[i])
        else:
            if type(l[i]) is NwObject:
                l[i] = object_dict[l[i].object_id()]
                print("tanslate in tuple:", l[i])

    return tuple(l)


def list_walker(l):
    if (l is None) or (not isinstance(l, list)):
        return
    for i in range(len(l)):
        if isinstance(l[i], list):
            list_walker(l[i])
        else:
            if type(l[i]) is NwObject:
                l[i] = object_dict[l[i].object_id()]
                print("tanslate in list:", l[i])


def dict_walker(d):
    if (d is None) or (type(d) is not dict):
        return
    for k, v in d.items():
        if isinstance(v, dict):
            dict_walker(d[k])
        elif isinstance(v, list):
            list_walker(d[k])
        else:
            if isinstance(v, NwObject):
                d[k] = object_dict[v.object_id()]
                print("tanslate in dict:", d[k])


def callback_constructor(callback_id, callback_param, param, mm, vm_id,
                         tf_queue, kvm_fd):
    class_id = param.base.class_id

    def request_callback(*args, **kwargs):
        global callback_stack

        writeback_result(vm_id, mm,
                         callback_param,
                         callback_param.param1,
                         args);
        writeback_result(vm_id, mm,
                         callback_param,
                         callback_param.param2,
                         kwargs);
        print("request callback id=%d" % callback_id)
        callback_param.base.object_id = callback_id
        param.base.done = STATUS_TASK_CALLBACK

        callback_stack.append({"callback_id":callback_id,
                               "param": param,
                               "callback_param": callback_param,
                               "deadline": time.time() + 20})

#        # spawn a new handler for TF APIs in the callback
#        t_cb_tf = threading.Thread(target = handler,
#                                   args = (tf_queue, kvm_fd, mm))
#        t_cb_tf.start()
#
#        # spin until done becomes RUNNING
#        spin_deadline = time.time() + 20
#        while time.time() < spin_deadline:
#            if param.base.done == STATUS_TASK_RUNNING:
#                print("callback finishes normally")
#                break
#            time.sleep(0.2)
#        print("callback loop exits")
#
#        # stop the callback thread
#        node = MINI_TASK_NODE()
#        node.vm_id = STOP_HANDLER
#        tf_queue.put(node)
#        t_cb_tf.join()
#        print("callback thread is shutdown")

        # handle in-callback APIs
        status = handler(tf_queue, kvm_fd, mm)

        callback_stack.pop()

        # validate and copy callback result
        if param.base.done != STATUS_TASK_RUNNING or \
           status != STATUS_CALLBACK_DONE:
            print("callback state error, state=%d, status=%d" %
                  param.base.done, status)
            return -1
        else:
            # TODO: update args and kwargs
            cb_ret = parse_param(vm_id, mm,
                                 callback_param,
                                 callback_param.ret_val1)
            print("receive callback result", cb_ret)
            if isinstance(cb_ret, list):
                list_walker(cb_ret)
            elif isinstance(cb_ret, dict):
                dict_walker(cb_ret)
            elif isinstance(cb_ret, tuple):
                cb_ret = tuple_walker(cb_ret)
            print("translated callback result", cb_ret)

            return cb_ret

    def tpu_estimator_callback(params):
        return request_callback(params)

    print("class_id=%d" % class_id)
    if class_id == TF_PY_TPU_TPU_ESTIMATOR:
        return tpu_estimator_callback
    else:
        return request_callback


def handler(queue, kvm_fd, mm):
    global object_dict
    global object_id
    global callback_stack

    global initialized
    if not initialized:
        callback_stack = []
        object_dict = dict()
        object_id = 1
        # TODO: forward logging or disable it in test
        tf.logging.set_verbosity(tf.logging.INFO)
        initialized = True
        print("handler is initialized")

    while True:
        task = None
        task = queue.get(block=True)

        while task is None:
            try:
                task = queue.get(block=True, timeout=5)
            except Queue.Empty:
                task = None
            if callback_stack:
                if time.time() > callback_stack[-1]["deadline"]:
                    print("callback failed deadline")
                    return STATUS_CALLBACK_TIMEOUT

        vm_id = task.vm_id
        if vm_id == STOP_HANDLER:
            break
        param = TF_PY_PARAM.from_buffer(mm, task.data_ptr)
        callback_param = TF_PY_PARAM.from_buffer(mm,
                task.data_ptr + param.base.callback_param_offset)
        print("retrieve [vm#%d] tensorflow task=%d cmd=%d, obj=%d, dstore=%lx, done=%d" %
              (task.vm_id, task.node_id, param.base.cmd_id,
               param.base.object_id, param.base.dstore_size,
               param.base.done))
        print("retrieve [vm#%d] callback node cmd=%d, obj=%d, dstore=%lx, done=%d" %
              (task.vm_id, callback_param.base.cmd_id,
               callback_param.base.object_id, callback_param.base.dstore_size,
               callback_param.base.done))

        cmd_id = param.base.cmd_id

        try:
            if cmd_id == TF_PY_NW_CALLBACK_DONE:
                param.base.done = STATUS_TASK_DONE
                ret = fcntl.ioctl(kvm_fd, IOCTL_KVM_NOTIFY_TASK_FINISHED, task.node_id)
                if ret < 0:
                    print("notify task completion failed: %d\n" % ret);
                if callback_stack and \
                   callback_stack[-1]["callback_id"] == param.base.object_id:
                    print("callback is finished")
                    return STATUS_CALLBACK_DONE
                else:
                    print("callback is error")
                    return STATUS_CALLBACK_ERROR

            if cmd_id == TF_PY_SESSION_INIT:
                print("SessionInit!!!")
                param1 = parse_param(vm_id, mm, param, param.param1)
                print(param1)
                sess = tf.Session(param1)

                # assign object_id
                object_dict[object_id] = sess
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_SESSION_ENTER:
                sess = object_dict[param.base.object_id]
                ctx_sess = sess.__enter__()
                if sess is ctx_sess:
                    pass
                else: # unlikely
                    print("unlikely to search for sess")
                    param.base.object_id = next(obj_id for obj_id, obj in
                            object_dict.items() if obj is ctx_sess)

            elif cmd_id == TF_PY_SESSION_EXIT:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)

                sess = object_dict[param.base.object_id]
                sess.__exit__(param1, param2, param3)

            elif cmd_id == TF_PY_SESSION_DEL:
                sess = object_dict[param.base.object_id]
                sess.__del__()

            # deprecated
            elif cmd_id == TF_PY_SESSION_RUN:
                sess = object_dict[param.base.object_id]
                param1 = parse_param(vm_id, mm, param, param.param1)

                if type(param1) == NwObject:
                    print("get NwObject=%d" % param1.object_id())
                    param1 = object_dict[param1.object_id()]
                    print(param1)

                ret_val = sess.run(param1)
                print(ret_val)

                writeback_result(vm_id, mm, param, param.ret_val1, ret_val);

            elif cmd_id == TF_PY_TPU_CLUSTER_RESOLVER_INIT:
                print("resloverInit!!!")
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                if param1 is None:
                    param1 = None
                if param2 is None:
                    param2 = None
                if param3 is None:
                    param3 = None
                print("TPUClusterResolver", param1, param2, param3)
                tpu_grpc = tf.contrib.cluster_resolver.TPUClusterResolver(
                        tpu=param1, zone=param2, project=param3)

                # assign object_id
                object_dict[object_id] = tpu_grpc
                param.base.object_id = object_id
                print("assign obj_id=%d" % object_id)
                object_id += 1

            # deprecated
            elif cmd_id == TF_PY_TPU_CLUSTER_RESOLVER_MASTER:
                # FIXED: use __getattr__
                print("master!!")
                tpu_grpc = object_dict[param.base.object_id]
                # FIXED: may have parameters
                tpu_grpc_url = tpu_grpc.master()

                # serialize return value
                writeback_result(vm_id, mm, param, param.ret_val1, tpu_grpc_url);

            elif cmd_id == TF_PY_TPU_INITIALIZE_SYSTEM:
                # TODO: may have parameters
                ts = tpu.initialize_system()

                object_dict[object_id] = ts
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_TPU_SHUTDOWN_SYSTEM:
                # TODO: may have parameters
                ts = tpu.shutdown_system()

                object_dict[object_id] = ts
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_GLOBAL_VARIABLES_INITIALIZER:
                # TODO: may have parameters
                ts = tf.global_variables_initializer()

                object_dict[object_id] = ts
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_ONES:
                print("param1 size=%ld,offset=%ld" %
                      (param.param1.size, param.param1.offset))
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                if param2 is None:
                    param2 = dtypes.float32
                print(param2)
                var = tf.ones(param1, param2)

                object_dict[object_id] = var
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_RANDOM_UNIFORM:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)
                param5 = parse_param(vm_id, mm, param, param.param5)
                param6 = parse_param(vm_id, mm, param, param.param6)
                if param2 is None:
                    param2 = 0
                if param4 is None:
                    param4 = dtypes.float32
                print(param1, param2, param3, param4)
                var = tf.random_uniform(param1, param2, param3, param4,
                                        param5, param6)

                object_dict[object_id] = var
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_TRANSPOSE:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)
                param1 = object_dict[param1.object_id()]
                if param3 is None:
                    param3 = "transpose"
                if param4 is None:
                    param4 = False
                print("transpose", param1, param2, param3, param4)
                var = tf.transpose(param1, param2, param3, param4)

                object_dict[object_id] = var
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_CAST:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param1 = object_dict[param1.object_id()]
                print("cast", param1, param2, param3)
                var = tf.cast(param1, param2, param3)

                object_dict[object_id] = var
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_EXPAND_DIMS:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)
                param1 = object_dict[param1.object_id()]
                print("expand_dims", param1, param2, param3, param4)
                var = tf.expand_dims(param1, param2, param3, param4)

                object_dict[object_id] = var
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_CONCAT:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param1 = object_dict[param1.object_id()]
                if param3 is None:
                    param3 = "concat"
                print("concat", param1, param2, param3)
                var = tf.concat(param1, param2, param3)

                object_dict[object_id] = var
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_EQUAL:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param1 = object_dict[param1.object_id()]
                print("equal", param1, param2, param3)
                if isinstance(param2, NwObject):
                    param2 = object_dict[param2.object_id()]
                result = tf.equal(param1, param2, param3)
                print(result)

                object_dict[object_id] = result
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_FIXED_LEN_FEATURE:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)

                feature = tf.FixedLenFeature(param1, param2, param3)
                print(feature)

                object_dict[object_id] = feature
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_VAR_LEN_FEATURE:
                param1 = parse_param(vm_id, mm, param, param.param1)

                feature = tf.VarLenFeature(param1)
                print(feature)

                object_dict[object_id] = feature
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_PARSE_SINGLE_EXAMPLE:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)
                print(param1, param2)

                # expand embedded NwObject
                if isinstance(param1, NwObject):
                    param1 = object_dict[param1.object_id()]
                dict_walker(param2)
                print("after translation", param1, param2)

                result = tf.parse_single_example(param1, param2, param3, param4)
                print(result)
                dict_mapper(result)
                print(result)
                writeback_result(vm_id, mm, param, param.ret_val1, result);

            elif cmd_id == TF_PY_CONTROL_FLOW_OPS_SWITCH:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)
                param1 = object_dict[param1.object_id()]
                param2 = object_dict[param2.object_id()]
                print("switch", param1, param2, param3, param4)
                result = control_flow_ops.switch(param1, param2, param3, param4)
                print(result)

                mapped_tuple = tuple_mapper(result, [0, 1])
                print(mapped_tuple)
                writeback_result(vm_id, mm, param, param.ret_val1, mapped_tuple);

            elif cmd_id == TF_PY_CONTROL_FLOW_OPS_MERGE:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param1 = object_dict[param1.object_id()]
                print("merge", param1, param2)
                list_walker(param1)
                print("merge-new", param1, param2)
                result = control_flow_ops.merge(param1, param2)
                print(result)

                mapped_tuple = tuple_mapper(result, [0])
                print(mapped_tuple)
                writeback_result(vm_id, mm, param, param.ret_val1, mapped_tuple);

            elif cmd_id == TF_PY_TPU_REWRITE:
                # TODO: may have parameters
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                # default parameter
                if param2 is None:
                    param2 = None
                # expand embedded NwObject
                list_walker(param2)
                func = tpu.rewrite(param1, param2)

                object_dict[object_id] = func
                param.base.object_id = object_id
                print("rewrite object_id=%d" % object_id)
                object_id += 1

            elif cmd_id == TF_PY_TPU_RUN_CONFIG:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)
                param5 = parse_param(vm_id, mm, param, param.param5)
                # default parameter
                if param1 is None:
                    param1 = None
                if param2 is None:
                    param2 = None
                if param3 is None:
                    param3 = None
                if param4 is None:
                    param4 = None

                # expand embedded NwObject
                param4 = object_dict[param4.object_id()]
                print(param4, param5)
                func = tpu.RunConfig(param1, param2, param3, param4,
                                     **param5)

                object_dict[object_id] = func
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_TPU_TPU_ESTIMATOR:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)
                param5 = parse_param(vm_id, mm, param, param.param5)
                param6 = parse_param(vm_id, mm, param, param.param6)
                param7 = parse_param(vm_id, mm, param, param.param7)
                param8 = parse_param(vm_id, mm, param, param.param8)
                param9 = parse_param(vm_id, mm, param, param.param9)
                param10 = parse_param(vm_id, mm, param, param.param10)
                param11 = parse_param(vm_id, mm, param, param.param11)
                param12 = parse_param(vm_id, mm, param, param.param12)
                # default parameter
                if param1 is None:
                    param1 = None
                if param2 is None:
                    param2 = None
                if param3 is None:
                    param3 = None
                if param4 is None:
                    param4 = None
                if param5 is None:
                    param5 = True
                if param6 is None:
                    param6 = None
                if param7 is None:
                    param7 = None
                if param8 is None:
                    param8 = None
                if param9 is None:
                    param9 = None
                if param10 is None:
                    param10 = True
                if param11 is None:
                    param11 = True
                if param12 is None:
                    param12 = None

                # expand embedded NwObject
                param3 = object_dict[param3.object_id()]
                print(param3)
                func = tpu.TPUEstimator(param1, param2, param3, param4,
                                        param5, param6, param7, param8,
                                        param9, param10, param11, param12)

                object_dict[object_id] = func
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_IMAGE_RESIZE_IMAGES:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)
                param5 = parse_param(vm_id, mm, param, param.param5)
                # default parameter
                if param3 is None:
                    param3 =ResizeMethod.BILINEAR
                if param4 is None:
                    param4 = False
                if param5 is None:
                    param5 = False

                # expand embedded NwObject
                param1 = object_dict[param1.object_id()]
                print(param1)
                img = tf.image.resize_images(param1, param2, param3, param4, param5)

                # TODO: it may return a float
                object_dict[object_id] = img
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_SLICE:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)

                # expand embedded NwObject
                print(param1, param2, param3)
                param1 = object_dict[param1.object_id()]
                ret = tf.slice(param1, param2, param3, param4)

                object_dict[object_id] = ret
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_SHAPE:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                if param3 is None:
                    param3 = dtypes.int32

                # expand embedded NwObject
                print(param1, param2, param3)
                param1 = object_dict[param1.object_id()]
                ret = tf.shape(param1, param2, param3)

                object_dict[object_id] = ret
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_IMAGE_SAMPLE_DISTORTED_BOUNDING_BOX:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)
                param5 = parse_param(vm_id, mm, param, param.param5)
                param6 = parse_param(vm_id, mm, param, param.param6)
                param7 = parse_param(vm_id, mm, param, param.param7)
                param8 = parse_param(vm_id, mm, param, param.param8)
                param9 = parse_param(vm_id, mm, param, param.param9)
                param10 = parse_param(vm_id, mm, param, param.param10)
                # default parameter
                if param5 is None:
                    param5 = 0.1

                print("sample_distorted_bounding_box", param1, param2)
                result = tf.image.sample_distorted_bounding_box(
                        param1, param2, param3, param4, param5, param6,
                        param7, param8, param9, param10)
                print(result)

                mapped_tuple = tuple_mapper(result, [0, 1, 2])
                print(mapped_tuple)
                writeback_result(vm_id, mm, param, param.ret_val1, mapped_tuple);

            elif cmd_id == TF_PY_IMAGE_DRAW_BOUNDING_BOXES:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)

                # expand embedded NwObject
                print(param1, param2, param3)
                param1 = object_dict[param1.object_id()]
                param2 = object_dict[param2.object_id()]
                ret = tf.image.draw_bounding_boxes(param1, param2, param3)

                object_dict[object_id] = ret
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_IMAGE_DECODE_JPEG:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)
                param5 = parse_param(vm_id, mm, param, param.param5)
                param6 = parse_param(vm_id, mm, param, param.param6)
                param7 = parse_param(vm_id, mm, param, param.param7)
                param8 = parse_param(vm_id, mm, param, param.param8)

                if param2 is None:
                    param2 = 0
                if param3 is None:
                    param3 = 1
                if param4 is None:
                    param4 = True
                if param5 is None:
                    param5 = False
                if param6 is None:
                    param6 = 1
                if param7 is None:
                    param7 = ""
                param1 = object_dict[param1.object_id()]
                img = tf.image.decode_jpeg(param1, param2, param3,
                        param4, param5, param6, param7, param8)

                object_dict[object_id] = img
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_IMAGE_CONVERT_IMAGE_DTYPE:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)
                param4 = parse_param(vm_id, mm, param, param.param4)

                # expand embedded NwObject
                print(param1, param2, param3)
                param1 = object_dict[param1.object_id()]
                if param3 is None:
                    param3 = False
                ret = tf.image.convert_image_dtype(param1, param2,
                        param3, param4)

                object_dict[object_id] = ret
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_DATA_DATASET_LIST_FILES:
                param1 = parse_param(vm_id, mm, param, param.param1)
                param2 = parse_param(vm_id, mm, param, param.param2)
                param3 = parse_param(vm_id, mm, param, param.param3)

                print(param1, param2, param3)
                if isinstance(param1, NwObject):
                    param1 = object_dict[oaram1.object_id()]
                ret = tf.data.Dataset.list_files(param1, param2, param3)

                object_dict[object_id] = ret
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_NW_OBJECT:
                print("nw_object!! id = %d" % param.base.object_id)
                obj = object_dict[param.base.object_id]
                name = parse_param(vm_id, mm, param, param.param1)
                args = parse_param(vm_id, mm, param, param.param2)
                kwargs = parse_param(vm_id, mm, param, param.param3)
                print("NwObject", obj, name, args, kwargs)

                # expand embedded NwObject
                args = list(args)
                list_walker(args)
                args = tuple(args)
                dict_walker(kwargs)
                print("after translation", obj, name, args, kwargs)

                # run
                result = getattr(obj, name)(*(args or []), **(kwargs or {}))
                param.base.object_id = -1
                param.ret_val1.size = 0
                print("analyze type", type(result), result)

                # TODO: go through tuple, dict or list
                if isinstance(result, tuple):
                    result = tuple_mapper(result, range(len(result)))
                if isinstance(result, dict):
                    dict_mapper(result)
                if isinstance(result, list):
                    list_mapper(result)

                # serialize return value
                if is_unpickleable_type(result) or \
                   pickle.pickles(result) is False:
                    object_dict[object_id] = result
                    param.base.object_id = object_id
                    object_id += 1

                elif result is not None:
                    writeback_result(vm_id, mm, param, param.ret_val1, result);

            elif cmd_id == TF_PY_NW_METHOD:
                # Reuse as callback

                #ins = parse_param(vm_id, mm, param, param.param1)
                #name = parse_param(vm_id, mm, param, param.param2)
                #print(ins, name)

                #method = getattr(ins, name)
                #print(method)
                #object_dict[object_id] = method

                cw = callback_constructor(object_id, callback_param,
                        param, mm, vm_id, queue, kvm_fd)
                object_dict[object_id] = cw
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_NW_CALLBACK_TEST:
                nw_func = parse_param(vm_id, mm, param, param.param1)
                print(nw_func, nw_func.object_id())
                func = object_dict[nw_func.object_id()]
                print("callback func", func)
                x = parse_param(vm_id, mm, param, param.param2)
                y = parse_param(vm_id, mm, param, param.param3)
                result = func(x, y)
                print(result)
                writeback_result(vm_id, mm, param, param.ret_val1, result);

            else:
                print("unsupported Tensorflow API")

        except Exception, error:
            param.base.done = STATUS_TASK_ERROR
            #mm.flush(task.data_ptr, sizeof(PARAM_BASE))

            print "fault: ", str(error)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            traceback.print_stack()

        print("finished [vm#%d] TF task %d cmd %d" %
              (task.vm_id, task.node_id, param.base.cmd_id))

        param.base.done = STATUS_TASK_DONE
        #mm.flush(task.data_ptr, sizeof(PARAM_BASE))
        #mm.flush(INVOKER_FIFO_SIZE + VGPU_DSTORE_SIZE * (vm_id - 1) +
        #         param.base.dstore_offset + param.ret_val1.offset,
        #         param.ret_val1.size)

        # notify hypervisor
        ret = fcntl.ioctl(kvm_fd, IOCTL_KVM_NOTIFY_TASK_FINISHED, task.node_id)
        if ret < 0:
            print("notify task completion failed: %d\n" % ret);
