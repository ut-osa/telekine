
6
inputPlaceholder*
shape
:*
dtype0
�
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"      *
dtype0
y
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *���*
dtype0
y
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *��?*
dtype0
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*
_class
loc:@dense/kernel
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel
�
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel
�
dense/kernel
VariableV2*
_class
loc:@dense/kernel*
dtype0*
	container *
shape
:*
shared_name 
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
U
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel
y
,dense/bias/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense/bias*
valueB:*
dtype0
n
"dense/bias/Initializer/zeros/ConstConst*
_class
loc:@dense/bias*
valueB
 *    *
dtype0
�
dense/bias/Initializer/zerosFill,dense/bias/Initializer/zeros/shape_as_tensor"dense/bias/Initializer/zeros/Const*
T0*
_class
loc:@dense/bias*

index_type0
y

dense/bias
VariableV2*
shape:*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container 
�
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(
O
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias
_
dense/MatMulMatMulinputdense/kernel/read*
transpose_a( *
transpose_b( *
T0
W
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*
data_formatNHWC
*
outputIdentitydense/BiasAdd*
T0
6
initNoOp^dense/kernel/Assign^dense/bias/Assign
8

save/ConstConst*
valueB Bmodel*
dtype0
]
save/SaveV2/tensor_namesConst*-
value$B"B
dense/biasBdense/kernel*
dtype0
K
save/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices
dense/biasdense/kernel*
dtypes
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
o
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*-
value$B"B
dense/biasBdense/kernel
]
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save/AssignAssign
dense/biassave/RestoreV2*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(
�
save/Assign_1Assigndense/kernelsave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
6
save/restore_allNoOp^save/Assign^save/Assign_1"