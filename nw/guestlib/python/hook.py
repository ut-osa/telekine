import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes

import os, sys, fcntl, atexit, types

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../include/common'))
from tensorflow_py import *
from devconf import *


init_vgpu()
fd = get_vgpu_fd()


# test

def NwTestCallback(func, x, y):
    param, cb_param = InitTFParamWithCallback(TF_PY_NW_CALLBACK_TEST)
    serialized_func = SerializeMethod(func)
    dump1, buf1 = SerializeParam(param, param.param1, serialized_func)
    dump2, buf2 = SerializeParam(param, param.param2, x)
    dump3, buf3 = SerializeParam(param, param.param3, y)

    buf = ReserveReturnValue(param, param.ret_val1)

    IoctlWrapper(param, cb_param)

    return pickle.loads(buf)


# tensorflow

def NwGlobalVariablesInitializer():
    param = InitTFParam(TF_PY_GLOBAL_VARIABLES_INITIALIZER)
    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tpu.initialize_system failed: %d\n" % ret);
        exit(-1);

    return NwObject(param.base.object_id)

tf.global_variables_initializer = NwGlobalVariablesInitializer

def NwOnes(shape, dtype=dtypes.float32, name=None):
    param = InitTFParam(TF_PY_ONES)
    dump1, buf1 = SerializeParam(param, param.param1, shape)
    if dtype != dtypes.float32:
        dump2, buf2 = SerializeParam(param, param.param2, dtype)
    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.ones failed: %d\n" % ret);
        exit(-1);

    print("ones object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.ones = NwOnes

def NwRandomUniform(shape,
                    minval=0,
                    maxval=None,
                    dtype=dtypes.float32,
                    seed=None,
                    name=None):
    param = InitTFParam(TF_PY_RANDOM_UNIFORM)

    dump1, buf1 = SerializeParam(param, param.param1, shape)
    if minval != 0:
        dump2, buf2 = SerializeParam(param, param.param2, minval)
    if maxval != None:
        dump3, buf3 = SerializeParam(param, param.param3, maxval)
    if dtype != dtypes.float32:
        dump4, buf4 = SerializeParam(param, param.param4, dtype)
    if seed != None:
        dump5, buf5 = SerializeParam(param, param.param5, seed)
    if name != None:
        dump6, buf6 = SerializeParam(param, param.param6, name)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.random_uniform failed: %d\n" % ret);
        exit(-1);

    print("random_uniform object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.random_uniform = NwRandomUniform

def NwTranspose(a, perm=None, name="transpose", conjugate=False):
    param = InitTFParam(TF_PY_TRANSPOSE)

    dump1, buf1 = SerializeParam(param, param.param1, a)
    if perm != None:
        dump2, buf2 = SerializeParam(param, param.param2, perm)
    if name != "transpose":
        dump3, buf3 = SerializeParam(param, param.param3, name)
    if conjugate != False:
        dump4, buf4 = SerializeParam(param, param.param4, conjugate)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.transpose failed: %d\n" % ret);
        exit(-1);

    print("transposed object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.transpose = NwTranspose

def NwCast(x, dtype, name=None):
    param = InitTFParam(TF_PY_CAST)

    dump1, buf1 = SerializeParam(param, param.param1, x)
    dump2, buf2 = SerializeParam(param, param.param2, dtype)
    if name != None:
        dump3, buf3 = SerializeParam(param, param.param3, name)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.cast failed: %d\n" % ret);
        exit(-1);

    print("cast object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.cast = NwCast

def NwExpandDims(input, axis=None, name=None, dim=None):
    param = InitTFParam(TF_PY_EXPAND_DIMS)

    dump1, buf1 = SerializeParam(param, param.param1, input)
    if axis != None:
        dump2, buf2 = SerializeParam(param, param.param2, axis)
    if name != None:
        dump3, buf3 = SerializeParam(param, param.param3, name)
    if dim != None:
        dump4, buf4 = SerializeParam(param, param.param4, dim)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.expand_dims failed: %d\n" % ret);
        exit(-1);

    print("cast object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.expand_dims = NwExpandDims

def NwConcat(values, axis, name="concat"):
    param = InitTFParam(TF_PY_CONCAT)

    dump1, buf1 = SerializeParam(param, param.param1, values)
    dump2, buf2 = SerializeParam(param, param.param2, axis)
    if name != "concat":
        dump3, buf3 = SerializeParam(param, param.param3, name)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.concat failed: %d\n" % ret);
        exit(-1);

    print("concat object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.concat = NwConcat

def NwEqual(x, y, name=None):
    param = InitTFParam(TF_PY_EQUAL)

    dump1, buf1 = SerializeParam(param, param.param1, x)
    dump2, buf2 = SerializeParam(param, param.param2, y)
    if name != None:
        dump3, buf3 = SerializeParam(param, param.param3, name)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.equal failed: %d\n" % ret);
        exit(-1);

    print("equal object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.equal = NwEqual

def NwSlice(input_, begin, size, name=None):
    param = InitTFParam(TF_PY_SLICE)

    dump1, buf1 = SerializeParam(param, param.param1, input_)
    dump2, buf2 = SerializeParam(param, param.param2, begin)
    dump3, buf3 = SerializeParam(param, param.param3, size)
    if name != None:
        dump4, buf4 = SerializeParam(param, param.param4, name)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.slice failed: %d\n" % ret);
        exit(-1);

    print("equal object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.slice = NwSlice

def NwShape(input_, name=None, out_type=dtypes.int32):
    param = InitTFParam(TF_PY_SHAPE)

    dump1, buf1 = SerializeParam(param, param.param1, input_)
    if name != None:
        dump2, buf2 = SerializeParam(param, param.param2, name)
    if out_type != dtypes.int32:
        dump3, buf3 = SerializeParam(param, param.param3, out_type)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.shape failed: %d\n" % ret);
        exit(-1);

    print("shape object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.shape = NwShape

def NwFixedLenFeature(shape, dtype, default_value=None):
    param = InitTFParam(TF_PY_FIXED_LEN_FEATURE)

    dump1, buf1 = SerializeParam(param, param.param1, shape)
    dump2, buf2 = SerializeParam(param, param.param2, dtype)
    if default_value != None:
        dump3, buf3 = SerializeParam(param, param.param3, default_value)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.FixedLenFeature failed: %d\n" % ret);
        exit(-1);

    print("FixedLenFeature object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.FixedLenFeature = NwFixedLenFeature

def NwVarLenFeature(dtype):
    param = InitTFParam(TF_PY_VAR_LEN_FEATURE)

    dump1, buf1 = SerializeParam(param, param.param1, dtype)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.VarLenFeature failed: %d\n" % ret);
        exit(-1);

    print("VarLenFeature object_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.VarLenFeature = NwVarLenFeature

def NwParseSingleExample(serialized, features, name=None,
                         example_names=None):
    param = InitTFParam(TF_PY_PARSE_SINGLE_EXAMPLE)

    dump1, buf1 = SerializeParam(param, param.param1, serialized)
    dump2, buf2 = SerializeParam(param, param.param2, features)
    if name != None:
        dump3, buf3 = SerializeParam(param, param.param3, name)
    if example_names != None:
        dump4, buf4 = SerializeParam(param, param.param4, example_names)

    buf = ReserveReturnValue(param, param.ret_val1)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tf.FixedLenFeature failed: %d\n" % ret);
        exit(-1);

    # returns a dict mapping feature keys to Tensor and SparseTensor
    # values.
    return pickle.loads(buf)

tf.parse_single_example = NwParseSingleExample


# tensorflow.contrib.tpu

def NwInitializeSystem(embedding_config=None, job=None):
    param = InitTFParam(TF_PY_TPU_INITIALIZE_SYSTEM)
    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tpu.initialize_system failed: %d\n" % ret);
        exit(-1);

    return NwObject(param.base.object_id)

tpu.initialize_system = NwInitializeSystem

'''
For a normal function, param.base.object_id returns the object id for the
remote object.
For a class method, param.base.object_id represents the object id for the
class object. The updated value represents the object id for the newly
created remote object.
'''
def NwShutdownSystem(job=None):
    param = InitTFParam(TF_PY_TPU_SHUTDOWN_SYSTEM)
    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tpu.initialize_system failed: %d\n" % ret);
        exit(-1);

    return NwObject(param.base.object_id)

tpu.shutdown_system = NwShutdownSystem

def NwRewrite(computation,
              inputs=None,
              infeed_queue=None,
              device_assignment=None,
              name=None):
    param = InitTFParam(TF_PY_TPU_REWRITE)

    dump1, buf1 = SerializeParam(param, param.param1, computation)
    if inputs != None:
        dump2, buf2 = SerializeParam(param, param.param2, inputs)
    print(param.param1.size, param.param2.size, param.base.dstore_size)
    print(buf1, dump1)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tpu.rewrite() failed: %d\n" % ret);
        exit(-1);

    print("rewrite object_id=%d" % param.base.object_id)
    print(buf1, dump1)
    return NwObject(param.base.object_id)

tpu.rewrite = NwRewrite

def NwRunConfig(tpu_config=None,
                evaluation_master=None,
                master=None,
                cluster=None,
                **kwargs):
    param = InitTFParam(TF_PY_TPU_RUN_CONFIG)

    if tpu_config != None:
        dump1, buf1 = SerializeParam(param, param.param1, tpu_config)
    if evaluation_master != None:
        dump2, buf2 = SerializeParam(param, param.param2, evaluation_master)
    if master != None:
        dump3, buf3 = SerializeParam(param, param.param3, master)
    # cluster is a NwObject
    # HACK: replace cluster with NwObject
    if cluster != None:
        dump4, buf4 = SerializeParam(param, param.param4, cluster)
    dump5, buf5 = SerializeParam(param, param.param5, kwargs)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tpu.rewrite() failed: %d\n" % ret);
        exit(-1);

    return NwObject(param.base.object_id)

tf.contrib.tpu.RunConfig = NwRunConfig

def NwTPUEstimator(model_fn=None,
                   model_dir=None,
                   config=None,
                   params=None,
                   use_tpu=True,
                   train_batch_size=None,
                   eval_batch_size=None,
                   predict_batch_size=None,
                   batch_axis=None,
                   eval_on_tpu=True,
                   export_to_tpu=True,
                   warm_start_from=None):
    param = InitTFParam(TF_PY_TPU_TPU_ESTIMATOR)

    if model_fn is not None:
        dump1, buf1 = SerializeParam(param, param.param1, model_fn)
    if model_dir is not None:
        dump2, buf2 = SerializeParam(param, param.param2, model_dir)
    # config is NwObject
    if config is not None:
        dump3, buf3 = SerializeParam(param, param.param3, config)
    if params is not None:
        dump4, buf4 = SerializeParam(param, param.param4, params)
    if use_tpu is not True:
        dump5, buf5 = SerializeParam(param, param.param5, use_tpu)
    if train_batch_size is not None:
        dump6, buf6 = SerializeParam(param, param.param6, train_batch_size)
    if eval_batch_size is not None:
        dump7, buf7 = SerializeParam(param, param.param7, eval_batch_size)
    if predict_batch_size is not None:
        dump8, buf8 = SerializeParam(param, param.param8, predict_batch_size)
    if batch_axis is not None:
        dump9, buf9 = SerializeParam(param, param.param9, batch_axis)
    if eval_on_tpu is not True:
        dump10, buf10 = SerializeParam(param, param.param10, eval_on_tpu)
    if export_to_tpu is not True:
        dump11, buf11 = SerializeParam(param, param.param11, export_to_tpu)
    if warm_start_from is not None:
        dump12, buf12 = SerializeParam(param, param.param12, warm_start_from)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd tpu.rewrite() failed: %d\n" % ret);
        exit(-1);

    return NwObject(param.base.object_id, class_id=TF_PY_TPU_TPU_ESTIMATOR)

tf.contrib.tpu.TPUEstimator = NwTPUEstimator


# Session

TfSession = tf.Session

class NwSession(TfSession):
    def __init__(self, target='', graph=None, config=None):
        print("[hook] inside Session Init")
        param = InitTFParam(TF_PY_SESSION_INIT)

        if target != '':
            dump1, buf1 = SerializeParam(param, param.param1, target)

        ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
        if ret < 0:
            print("ioctl_tf_cmd Session() failed: %d\n" % ret);
            exit(-1);

        self._object_id = param.base.object_id
        if (self._object_id < 0):
            return None

    def __enter__(self):
        param = InitTFParam(TF_PY_SESSION_ENTER)
        param.base.object_id = self._object_id
        ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
        if ret < 0:
            print("ioctl_tf_cmd Session.__enter__ failed: %d\n" % ret);
            exit(-1);
        if self._object_id != param.base.object_id:
            print("object IDs mismatch")
            self._object_id = param.base.object_id
        return self

    def __exit__(self, exec_type, exec_value, exec_tb):
        print(exec_type, exec_value, exec_tb)
        return
        param = InitTFParam(TF_PY_SESSION_INIT)
        param.base.object_id = self._object_id

        dump1, buf1 = SerializeParam(param, param.param1, exec_type)
        dump2, buf2 = SerializeParam(param, param.param2, exec_value)
        dump3, buf3 = SerializeParam(param, param.param3, exec_tb)

        ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
        if ret < 0:
            print("ioctl_tf_cmd Session.__exit__ failed: %d\n" % ret);
            exit(-1);

    def run(self, fetches, feed_dict=None, options=None,
            run_metadata=None):
        print("run!")
        param = InitTFParam(TF_PY_SESSION_RUN)
        param.base.object_id = self._object_id

        dump1, buf1 = SerializeParam(param, param.param1, fetches)

        buf = ReserveReturnValue(param, param.ret_val1)

        ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
        if ret < 0:
            print("ioctl_tf_cmd Session.run failed: %d\n" % ret);
            exit(-1);

        return pickle.loads(buf)

    def __del__(self):
        param = InitTFParam(TF_PY_SESSION_DEL)
        param.base.object_id = self._object_id
        ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
        if ret < 0:
            print("ioctl_tf_cmd Session.__del__ failed: %d\n" % ret);
            exit(-1);

def NwSession2(target='', graph=None, config=None):
    param = InitTFParam(TF_PY_SESSION_INIT)

    dump1, buf1 = SerializeParam(param, param.param1, target)
    if graph is not None:
        dump2, buf2 = SerializeParam(param, param.param2, graph)
    if config is not None:
        dump3, buf3 = SerializeParam(param, param.param3, config)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd SessionInit() failed: %d\n" % ret);
        exit(-1);

    return NwObject(param.base.object_id)

tf.Session = NwSession2


# TPUClusterResolver

def NwTPUClusterResolver(tpu=None,
                         zone=None,
                         project=None,
                         job_name='worker',
                         coordinator_name=None,
                         coordinator_address=None,
                         credentials='default',
                         service=None,
                         discovery_url=None):
    print("[hook] inside TPUClusterResolver")
    param = InitTFParam(TF_PY_TPU_CLUSTER_RESOLVER_INIT)

    if tpu is not None:
        dump1, buf1 = SerializeParam(param, param.param1, tpu)
    if zone is not None:
        dump2, buf2 = SerializeParam(param, param.param2, zone)
    if project is not None:
        dump3, buf3 = SerializeParam(param, param.param3, project)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd TPUClusterResolver() failed: %d\n" % ret);
        exit(-1);

    print("obj_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.contrib.cluster_resolver.TPUClusterResolver = NwTPUClusterResolver
TPUClusterResolver = NwTPUClusterResolver


# control_flow_ops

def NwControlFlowOpsSwitch(data, pred, dtype=None, name=None):
    param = InitTFParam(TF_PY_CONTROL_FLOW_OPS_SWITCH)

    dump1, buf1 = SerializeParam(param, param.param1, data)
    dump2, buf2 = SerializeParam(param, param.param2, pred)
    if dtype is not None:
        dump3, buf3 = SerializeParam(param, param.param3, dtype)
    if name is not None:
        dump4, buf4 = SerializeParam(param, param.param4, name)

    # returns a tuple (NwObject, NwObject)
    buf = ReserveReturnValue(param, param.ret_val1)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd control_flow_ops.switch failed: %d\n" % ret);
        exit(-1);

    return pickle.loads(buf)

control_flow_ops.switch = NwControlFlowOpsSwitch

def NwControlFlowOpsMerge(inputs, name=None):
    param = InitTFParam(TF_PY_CONTROL_FLOW_OPS_MERGE)

    dump1, buf1 = SerializeParam(param, param.param1, inputs)
    if name is not None:
        dump2, buf2 = SerializeParam(param, param.param2, name)

    # returns a tuple (NwObject, index)
    buf = ReserveReturnValue(param, param.ret_val1)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd control_flow_ops.switch failed: %d\n" % ret);
        exit(-1);

    return pickle.loads(buf)

control_flow_ops.merge = NwControlFlowOpsMerge


# tf.image

def NwImageResizeImages(images,
                        size,
                        method=tf.image.ResizeMethod.BILINEAR,
                        align_corners=False,
                        preserve_aspect_ratio=False):
    param = InitTFParam(TF_PY_IMAGE_RESIZE_IMAGES)

    dump1, buf1 = SerializeParam(param, param.param1, images)
    dump2, buf2 = SerializeParam(param, param.param2, size)
    if method != tf.image.ResizeMethod.BILINEAR:
        dump3, buf3 = SerializeParam(param, param.param3, method)
    if align_corners != False:
        dump4, buf4 = SerializeParam(param, param.param4, align_corners)
    if preserve_aspect_ratio != False:
        dump5, buf5 = SerializeParam(param, param.param5, preserve_aspect_ratio)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd image.resize_image() failed: %d\n" % ret);
        exit(-1);

    print("obj_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.image.resize_images = NwImageResizeImages

def NwImageSampleDistortedBoundingBox(image_size,
                                      bounding_boxes,
                                      seed=None,
                                      seed2=None,
                                      min_object_covered=0.1,
                                      aspect_ratio_range=None,
                                      area_range=None,
                                      max_attempts=None,
                                      use_image_if_no_bounding_boxes=None,
                                      name=None):
    param = InitTFParam(TF_PY_IMAGE_SAMPLE_DISTORTED_BOUNDING_BOX)

    dump1, buf1 = SerializeParam(param, param.param1, image_size)
    dump2, buf2 = SerializeParam(param, param.param2, bounding_boxes)
    if seed != None:
        dump3, buf3 = SerializeParam(param, param.param3, seed)
    if seed2 != None:
        dump4, buf4 = SerializeParam(param, param.param4, seed2)
    if min_object_covered != 0.1:
        dump5, buf5 = SerializeParam(param, param.param5, min_object_covered)
    if aspect_ratio_range != None:
        dump6, buf6 = SerializeParam(param, param.param6, aspect_ratio_range)
    if area_range != None:
        dump7, buf7 = SerializeParam(param, param.param7, area_range)
    if max_attempts != None:
        dump8, buf8 = SerializeParam(param, param.param8, max_attempts)
    if use_image_if_no_bounding_boxes != None:
        dump9, buf9 = SerializeParam(param, param.param9, use_image_if_no_bounding_boxes)
    if name != None:
        dump10, buf10 = SerializeParam(param, param.param10, name)

    # returns a tuple (NwObject, NwObject, NwObject)
    buf = ReserveReturnValue(param, param.ret_val1)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd image.sample_distorted_bounding_box failed: %d\n" % ret);
        exit(-1);

    return pickle.loads(buf)

tf.image.sample_distorted_bounding_box = NwImageSampleDistortedBoundingBox

def NwImageDrawBoundingBoxes(images, boxes, name=None):
    param = InitTFParam(TF_PY_IMAGE_DRAW_BOUNDING_BOXES)

    dump1, buf1 = SerializeParam(param, param.param1, images)
    dump2, buf2 = SerializeParam(param, param.param2, boxes)
    if name != None:
        dump3, buf3 = SerializeParam(param, param.param3, name)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd image.draw_bounding_boxes() failed: %d\n" % ret);
        exit(-1);

    print("bounding boxes obj_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.image.draw_bounding_boxes = NwImageDrawBoundingBoxes

def NwImageDecodeJpeg(contents,
                      channels=0,
                      ratio=1,
                      fancy_upscaling=True,
                      try_recover_truncated=False,
                      acceptable_fraction=1,
                      dct_method="",
                      name=None):
    param = InitTFParam(TF_PY_IMAGE_DECODE_JPEG)

    dump1, buf1 = SerializeParam(param, param.param1, contents)
    if channels != 0:
        dump2, buf2 = SerializeParam(param, param.param2, channels)
    if ratio != 1:
        dump3, buf3 = SerializeParam(param, param.param3, ratio)
    if fancy_upscaling != True:
        dump4, buf4 = SerializeParam(param, param.param4, fancy_upscaling)
    if try_recover_truncated != False:
        dump5, buf5 = SerializeParam(param, param.param5, try_recover_truncated)
    if acceptable_fraction != 1:
        dump6, buf6 = SerializeParam(param, param.param6, acceptable_fraction)
    if dct_method != "":
        dump7, buf7 = SerializeParam(param, param.param7, dct_method)
    if name != None:
        dump8, buf8 = SerializeParam(param, param.param8, name)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd image.decode_jpeg failed: %d\n" % ret);
        exit(-1);

    print("decoded jpeg obj_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.image.decode_jpeg = NwImageDecodeJpeg

def NwImageConvertImageDtype(image, dtype, saturate=False, name=None):
    param = InitTFParam(TF_PY_IMAGE_CONVERT_IMAGE_DTYPE)

    dump1, buf1 = SerializeParam(param, param.param1, images)
    dump2, buf2 = SerializeParam(param, param.param2, dtype)
    if saturate != False:
        dump3, buf3 = SerializeParam(param, param.param3, saturate)
    if name != None:
        dump4, buf4 = SerializeParam(param, param.param4, name)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd image.convert_image_dtype failed: %d\n" % ret);
        exit(-1);

    print("convert_image_dtype obj_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.image.convert_image_dtype = NwImageConvertImageDtype


# tf.data.Dataset

@staticmethod
def NwDataDatasetListFiles(file_pattern, shuffle=None, seed=None):
    print("enter NwDataDatasetListFiles")
    param = InitTFParam(TF_PY_DATA_DATASET_LIST_FILES)

    dump1, buf1 = SerializeParam(param, param.param1, file_pattern)
    if shuffle != None:
        dump2, buf2 = SerializeParam(param, param.param2, shuffle)
    if seed != None:
        dump3, buf3 = SerializeParam(param, param.param3, seed)

    ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
    if ret < 0:
        print("ioctl_tf_cmd data.Dataset.list_files failed: %d\n" % ret);
        exit(-1);

    print("obj_id=%d" % param.base.object_id)
    return NwObject(param.base.object_id)

tf.data.Dataset.list_files = NwDataDatasetListFiles


# Common

@atexit.register
def close_fd():
    fd.close
    print("[hook] fd is closed")
