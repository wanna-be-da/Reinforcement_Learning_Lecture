Ã
Ì¢
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.22v2.8.2-0-g2ea19cbb5758¬
|
dense_119/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_119/kernel
u
$dense_119/kernel/Read/ReadVariableOpReadVariableOpdense_119/kernel*
_output_shapes

:*
dtype0
t
dense_119/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_119/bias
m
"dense_119/bias/Read/ReadVariableOpReadVariableOpdense_119/bias*
_output_shapes
:*
dtype0
|
dense_117/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*!
shared_namedense_117/kernel
u
$dense_117/kernel/Read/ReadVariableOpReadVariableOpdense_117/kernel*
_output_shapes

:

*
dtype0
t
dense_117/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_117/bias
m
"dense_117/bias/Read/ReadVariableOpReadVariableOpdense_117/bias*
_output_shapes
:
*
dtype0
|
dense_118/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_namedense_118/kernel
u
$dense_118/kernel/Read/ReadVariableOpReadVariableOpdense_118/kernel*
_output_shapes

:
*
dtype0
t
dense_118/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_118/bias
m
"dense_118/bias/Read/ReadVariableOpReadVariableOpdense_118/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ð
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*«
value¡B B
ý
input_layer
hidden_layers
output_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
call

signatures*
* 

0
1*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 
* 

 serving_default* 
¦

kernel
bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
¦

kernel
bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*
XR
VARIABLE_VALUEdense_119/kernel.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_119/bias,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
PJ
VARIABLE_VALUEdense_117/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_117/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_118/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_118/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*
* 
* 
* 
* 

0
1*

0
1*
* 

2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

¤
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_117/kerneldense_117/biasdense_118/kerneldense_118/biasdense_119/kerneldense_119/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_513882
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ÿ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_119/kernel/Read/ReadVariableOp"dense_119/bias/Read/ReadVariableOp$dense_117/kernel/Read/ReadVariableOp"dense_117/bias/Read/ReadVariableOp$dense_118/kernel/Read/ReadVariableOp"dense_118/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_513982

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_119/kerneldense_119/biasdense_117/kerneldense_117/biasdense_118/kerneldense_118/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_514010þý


ö
E__inference_dense_118_layer_call_and_return_conditional_losses_513461

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs


ö
E__inference_dense_118_layer_call_and_return_conditional_losses_513941

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Å
þ
$__inference_signature_wrapper_513882
input_1
unknown:


	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_513426o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ë
ä
__inference_call_513815

states:
(dense_117_matmul_readvariableop_resource:

7
)dense_117_biasadd_readvariableop_resource:
:
(dense_118_matmul_readvariableop_resource:
7
)dense_118_biasadd_readvariableop_resource::
(dense_119_matmul_readvariableop_resource:7
)dense_119_biasadd_readvariableop_resource:
identity¢ dense_117/BiasAdd/ReadVariableOp¢dense_117/MatMul/ReadVariableOp¢ dense_118/BiasAdd/ReadVariableOp¢dense_118/MatMul/ReadVariableOp¢ dense_119/BiasAdd/ReadVariableOp¢dense_119/MatMul/ReadVariableOp
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0t
dense_117/MatMulMatMulstates'dense_117/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
[
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*
_output_shapes

:

dense_118/MatMul/ReadVariableOpReadVariableOp(dense_118_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_118/MatMulMatMuldense_117/Relu:activations:0'dense_118/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_118/BiasAdd/ReadVariableOpReadVariableOp)dense_118_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_118/BiasAddBiasAdddense_118/MatMul:product:0(dense_118/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:[
dense_118/ReluReludense_118/BiasAdd:output:0*
T0*
_output_shapes

:
dense_119/MatMul/ReadVariableOpReadVariableOp(dense_119_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_119/MatMulMatMuldense_118/Relu:activations:0'dense_119/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_119/BiasAdd/ReadVariableOpReadVariableOp)dense_119_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_119/BiasAddBiasAdddense_119/MatMul:product:0(dense_119/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_119/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp!^dense_118/BiasAdd/ReadVariableOp ^dense_118/MatMul/ReadVariableOp!^dense_119/BiasAdd/ReadVariableOp ^dense_119/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2D
 dense_118/BiasAdd/ReadVariableOp dense_118/BiasAdd/ReadVariableOp2B
dense_118/MatMul/ReadVariableOpdense_118/MatMul/ReadVariableOp2D
 dense_119/BiasAdd/ReadVariableOp dense_119/BiasAdd/ReadVariableOp2B
dense_119/MatMul/ReadVariableOpdense_119/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates
õ

-__inference_qfunction_39_layer_call_fn_513599
input_1
unknown:


	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513567o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ë
ä
__inference_call_513743

states:
(dense_117_matmul_readvariableop_resource:

7
)dense_117_biasadd_readvariableop_resource:
:
(dense_118_matmul_readvariableop_resource:
7
)dense_118_biasadd_readvariableop_resource::
(dense_119_matmul_readvariableop_resource:7
)dense_119_biasadd_readvariableop_resource:
identity¢ dense_117/BiasAdd/ReadVariableOp¢dense_117/MatMul/ReadVariableOp¢ dense_118/BiasAdd/ReadVariableOp¢dense_118/MatMul/ReadVariableOp¢ dense_119/BiasAdd/ReadVariableOp¢dense_119/MatMul/ReadVariableOp
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0t
dense_117/MatMulMatMulstates'dense_117/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
[
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*
_output_shapes

:

dense_118/MatMul/ReadVariableOpReadVariableOp(dense_118_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_118/MatMulMatMuldense_117/Relu:activations:0'dense_118/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_118/BiasAdd/ReadVariableOpReadVariableOp)dense_118_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_118/BiasAddBiasAdddense_118/MatMul:product:0(dense_118/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:[
dense_118/ReluReludense_118/BiasAdd:output:0*
T0*
_output_shapes

:
dense_119/MatMul/ReadVariableOpReadVariableOp(dense_119_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_119/MatMulMatMuldense_118/Relu:activations:0'dense_119/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_119/BiasAdd/ReadVariableOpReadVariableOp)dense_119_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_119/BiasAddBiasAdddense_119/MatMul:product:0(dense_119/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_119/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp!^dense_118/BiasAdd/ReadVariableOp ^dense_118/MatMul/ReadVariableOp!^dense_119/BiasAdd/ReadVariableOp ^dense_119/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2D
 dense_118/BiasAdd/ReadVariableOp dense_118/BiasAdd/ReadVariableOp2B
dense_118/MatMul/ReadVariableOpdense_118/MatMul/ReadVariableOp2D
 dense_119/BiasAdd/ReadVariableOp dense_119/BiasAdd/ReadVariableOp2B
dense_119/MatMul/ReadVariableOpdense_119/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates
È	
ö
E__inference_dense_119_layer_call_and_return_conditional_losses_513477

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Æ
!__inference__wrapped_model_513426
input_1%
qfunction_39_513412:

!
qfunction_39_513414:
%
qfunction_39_513416:
!
qfunction_39_513418:%
qfunction_39_513420:!
qfunction_39_513422:
identity¢$qfunction_39/StatefulPartitionedCall¬
$qfunction_39/StatefulPartitionedCallStatefulPartitionedCallinput_1qfunction_39_513412qfunction_39_513414qfunction_39_513416qfunction_39_513418qfunction_39_513420qfunction_39_513422*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_call_513411|
IdentityIdentity-qfunction_39/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
NoOpNoOp%^qfunction_39/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2L
$qfunction_39/StatefulPartitionedCall$qfunction_39/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1


ö
E__inference_dense_117_layer_call_and_return_conditional_losses_513444

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ë
ä
__inference_call_513791

states:
(dense_117_matmul_readvariableop_resource:

7
)dense_117_biasadd_readvariableop_resource:
:
(dense_118_matmul_readvariableop_resource:
7
)dense_118_biasadd_readvariableop_resource::
(dense_119_matmul_readvariableop_resource:7
)dense_119_biasadd_readvariableop_resource:
identity¢ dense_117/BiasAdd/ReadVariableOp¢dense_117/MatMul/ReadVariableOp¢ dense_118/BiasAdd/ReadVariableOp¢dense_118/MatMul/ReadVariableOp¢ dense_119/BiasAdd/ReadVariableOp¢dense_119/MatMul/ReadVariableOp
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0t
dense_117/MatMulMatMulstates'dense_117/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
[
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*
_output_shapes

:

dense_118/MatMul/ReadVariableOpReadVariableOp(dense_118_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_118/MatMulMatMuldense_117/Relu:activations:0'dense_118/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_118/BiasAdd/ReadVariableOpReadVariableOp)dense_118_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_118/BiasAddBiasAdddense_118/MatMul:product:0(dense_118/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:[
dense_118/ReluReludense_118/BiasAdd:output:0*
T0*
_output_shapes

:
dense_119/MatMul/ReadVariableOpReadVariableOp(dense_119_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_119/MatMulMatMuldense_118/Relu:activations:0'dense_119/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_119/BiasAdd/ReadVariableOpReadVariableOp)dense_119_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_119/BiasAddBiasAdddense_119/MatMul:product:0(dense_119/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_119/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp!^dense_118/BiasAdd/ReadVariableOp ^dense_118/MatMul/ReadVariableOp!^dense_119/BiasAdd/ReadVariableOp ^dense_119/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2D
 dense_118/BiasAdd/ReadVariableOp dense_118/BiasAdd/ReadVariableOp2B
dense_118/MatMul/ReadVariableOpdense_118/MatMul/ReadVariableOp2D
 dense_119/BiasAdd/ReadVariableOp dense_119/BiasAdd/ReadVariableOp2B
dense_119/MatMul/ReadVariableOpdense_119/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates
ò

-__inference_qfunction_39_layer_call_fn_513654

states
unknown:


	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallstatesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513484o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates


ö
E__inference_dense_117_layer_call_and_return_conditional_losses_513921

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ò

__inference__traced_save_513982
file_prefix/
+savev2_dense_119_kernel_read_readvariableop-
)savev2_dense_119_bias_read_readvariableop/
+savev2_dense_117_kernel_read_readvariableop-
)savev2_dense_117_bias_read_readvariableop/
+savev2_dense_118_kernel_read_readvariableop-
)savev2_dense_118_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*±
value§B¤B.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B ¾
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_119_kernel_read_readvariableop)savev2_dense_119_bias_read_readvariableop+savev2_dense_117_kernel_read_readvariableop)savev2_dense_117_bias_read_readvariableop+savev2_dense_118_kernel_read_readvariableop)savev2_dense_118_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*G
_input_shapes6
4: :::

:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: 
Ä

*__inference_dense_117_layer_call_fn_513910

inputs
unknown:


	unknown_0:

identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_117_layer_call_and_return_conditional_losses_513444o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Î
ä
__inference_call_513863

states:
(dense_117_matmul_readvariableop_resource:

7
)dense_117_biasadd_readvariableop_resource:
:
(dense_118_matmul_readvariableop_resource:
7
)dense_118_biasadd_readvariableop_resource::
(dense_119_matmul_readvariableop_resource:7
)dense_119_biasadd_readvariableop_resource:
identity¢ dense_117/BiasAdd/ReadVariableOp¢dense_117/MatMul/ReadVariableOp¢ dense_118/BiasAdd/ReadVariableOp¢dense_118/MatMul/ReadVariableOp¢ dense_119/BiasAdd/ReadVariableOp¢dense_119/MatMul/ReadVariableOp
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0}
dense_117/MatMulMatMulstates'dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_118/MatMul/ReadVariableOpReadVariableOp(dense_118_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_118/MatMulMatMuldense_117/Relu:activations:0'dense_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_118/BiasAdd/ReadVariableOpReadVariableOp)dense_118_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_118/BiasAddBiasAdddense_118/MatMul:product:0(dense_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_118/ReluReludense_118/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_119/MatMul/ReadVariableOpReadVariableOp(dense_119_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_119/MatMulMatMuldense_118/Relu:activations:0'dense_119/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_119/BiasAdd/ReadVariableOpReadVariableOp)dense_119_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_119/BiasAddBiasAdddense_119/MatMul:product:0(dense_119/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_119/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp!^dense_118/BiasAdd/ReadVariableOp ^dense_118/MatMul/ReadVariableOp!^dense_119/BiasAdd/ReadVariableOp ^dense_119/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2D
 dense_118/BiasAdd/ReadVariableOp dense_118/BiasAdd/ReadVariableOp2B
dense_118/MatMul/ReadVariableOpdense_118/MatMul/ReadVariableOp2D
 dense_119/BiasAdd/ReadVariableOp dense_119/BiasAdd/ReadVariableOp2B
dense_119/MatMul/ReadVariableOpdense_119/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
¥

H__inference_qfunction_39_layer_call_and_return_conditional_losses_513484

states"
dense_117_513445:


dense_117_513447:
"
dense_118_513462:

dense_118_513464:"
dense_119_513478:
dense_119_513480:
identity¢!dense_117/StatefulPartitionedCall¢!dense_118/StatefulPartitionedCall¢!dense_119/StatefulPartitionedCallô
!dense_117/StatefulPartitionedCallStatefulPartitionedCallstatesdense_117_513445dense_117_513447*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_117_layer_call_and_return_conditional_losses_513444
!dense_118/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0dense_118_513462dense_118_513464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_118_layer_call_and_return_conditional_losses_513461
!dense_119/StatefulPartitionedCallStatefulPartitionedCall*dense_118/StatefulPartitionedCall:output:0dense_119_513478dense_119_513480*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_119_layer_call_and_return_conditional_losses_513477y
IdentityIdentity*dense_119/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^dense_117/StatefulPartitionedCall"^dense_118/StatefulPartitionedCall"^dense_119/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2F
!dense_118/StatefulPartitionedCall!dense_118/StatefulPartitionedCall2F
!dense_119/StatefulPartitionedCall!dense_119/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
ë
ä
__inference_call_513839

states:
(dense_117_matmul_readvariableop_resource:

7
)dense_117_biasadd_readvariableop_resource:
:
(dense_118_matmul_readvariableop_resource:
7
)dense_118_biasadd_readvariableop_resource::
(dense_119_matmul_readvariableop_resource:7
)dense_119_biasadd_readvariableop_resource:
identity¢ dense_117/BiasAdd/ReadVariableOp¢dense_117/MatMul/ReadVariableOp¢ dense_118/BiasAdd/ReadVariableOp¢dense_118/MatMul/ReadVariableOp¢ dense_119/BiasAdd/ReadVariableOp¢dense_119/MatMul/ReadVariableOp
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0t
dense_117/MatMulMatMulstates'dense_117/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
[
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*
_output_shapes

:

dense_118/MatMul/ReadVariableOpReadVariableOp(dense_118_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_118/MatMulMatMuldense_117/Relu:activations:0'dense_118/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_118/BiasAdd/ReadVariableOpReadVariableOp)dense_118_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_118/BiasAddBiasAdddense_118/MatMul:product:0(dense_118/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:[
dense_118/ReluReludense_118/BiasAdd:output:0*
T0*
_output_shapes

:
dense_119/MatMul/ReadVariableOpReadVariableOp(dense_119_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_119/MatMulMatMuldense_118/Relu:activations:0'dense_119/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_119/BiasAdd/ReadVariableOpReadVariableOp)dense_119_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_119/BiasAddBiasAdddense_119/MatMul:product:0(dense_119/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_119/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp!^dense_118/BiasAdd/ReadVariableOp ^dense_118/MatMul/ReadVariableOp!^dense_119/BiasAdd/ReadVariableOp ^dense_119/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2D
 dense_118/BiasAdd/ReadVariableOp dense_118/BiasAdd/ReadVariableOp2B
dense_118/MatMul/ReadVariableOpdense_118/MatMul/ReadVariableOp2D
 dense_119/BiasAdd/ReadVariableOp dense_119/BiasAdd/ReadVariableOp2B
dense_119/MatMul/ReadVariableOpdense_119/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates
Ä

*__inference_dense_118_layer_call_fn_513930

inputs
unknown:

	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_118_layer_call_and_return_conditional_losses_513461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
È	
ö
E__inference_dense_119_layer_call_and_return_conditional_losses_513901

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
 
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513618
input_1"
dense_117_513602:


dense_117_513604:
"
dense_118_513607:

dense_118_513609:"
dense_119_513612:
dense_119_513614:
identity¢!dense_117/StatefulPartitionedCall¢!dense_118/StatefulPartitionedCall¢!dense_119/StatefulPartitionedCallõ
!dense_117/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_117_513602dense_117_513604*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_117_layer_call_and_return_conditional_losses_513444
!dense_118/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0dense_118_513607dense_118_513609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_118_layer_call_and_return_conditional_losses_513461
!dense_119/StatefulPartitionedCallStatefulPartitionedCall*dense_118/StatefulPartitionedCall:output:0dense_119_513612dense_119_513614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_119_layer_call_and_return_conditional_losses_513477y
IdentityIdentity*dense_119/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^dense_117/StatefulPartitionedCall"^dense_118/StatefulPartitionedCall"^dense_119/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2F
!dense_118/StatefulPartitionedCall!dense_118/StatefulPartitionedCall2F
!dense_119/StatefulPartitionedCall!dense_119/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ò

-__inference_qfunction_39_layer_call_fn_513671

states
unknown:


	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallstatesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513567o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
Î
ä
__inference_call_513411

states:
(dense_117_matmul_readvariableop_resource:

7
)dense_117_biasadd_readvariableop_resource:
:
(dense_118_matmul_readvariableop_resource:
7
)dense_118_biasadd_readvariableop_resource::
(dense_119_matmul_readvariableop_resource:7
)dense_119_biasadd_readvariableop_resource:
identity¢ dense_117/BiasAdd/ReadVariableOp¢dense_117/MatMul/ReadVariableOp¢ dense_118/BiasAdd/ReadVariableOp¢dense_118/MatMul/ReadVariableOp¢ dense_119/BiasAdd/ReadVariableOp¢dense_119/MatMul/ReadVariableOp
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0}
dense_117/MatMulMatMulstates'dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_118/MatMul/ReadVariableOpReadVariableOp(dense_118_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_118/MatMulMatMuldense_117/Relu:activations:0'dense_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_118/BiasAdd/ReadVariableOpReadVariableOp)dense_118_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_118/BiasAddBiasAdddense_118/MatMul:product:0(dense_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_118/ReluReludense_118/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_119/MatMul/ReadVariableOpReadVariableOp(dense_119_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_119/MatMulMatMuldense_118/Relu:activations:0'dense_119/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_119/BiasAdd/ReadVariableOpReadVariableOp)dense_119_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_119/BiasAddBiasAdddense_119/MatMul:product:0(dense_119/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_119/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp!^dense_118/BiasAdd/ReadVariableOp ^dense_118/MatMul/ReadVariableOp!^dense_119/BiasAdd/ReadVariableOp ^dense_119/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2D
 dense_118/BiasAdd/ReadVariableOp dense_118/BiasAdd/ReadVariableOp2B
dense_118/MatMul/ReadVariableOpdense_118/MatMul/ReadVariableOp2D
 dense_119/BiasAdd/ReadVariableOp dense_119/BiasAdd/ReadVariableOp2B
dense_119/MatMul/ReadVariableOpdense_119/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
õ

-__inference_qfunction_39_layer_call_fn_513499
input_1
unknown:


	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513484o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ÿ

H__inference_qfunction_39_layer_call_and_return_conditional_losses_513719

states:
(dense_117_matmul_readvariableop_resource:

7
)dense_117_biasadd_readvariableop_resource:
:
(dense_118_matmul_readvariableop_resource:
7
)dense_118_biasadd_readvariableop_resource::
(dense_119_matmul_readvariableop_resource:7
)dense_119_biasadd_readvariableop_resource:
identity¢ dense_117/BiasAdd/ReadVariableOp¢dense_117/MatMul/ReadVariableOp¢ dense_118/BiasAdd/ReadVariableOp¢dense_118/MatMul/ReadVariableOp¢ dense_119/BiasAdd/ReadVariableOp¢dense_119/MatMul/ReadVariableOp
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0}
dense_117/MatMulMatMulstates'dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_118/MatMul/ReadVariableOpReadVariableOp(dense_118_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_118/MatMulMatMuldense_117/Relu:activations:0'dense_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_118/BiasAdd/ReadVariableOpReadVariableOp)dense_118_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_118/BiasAddBiasAdddense_118/MatMul:product:0(dense_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_118/ReluReludense_118/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_119/MatMul/ReadVariableOpReadVariableOp(dense_119_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_119/MatMulMatMuldense_118/Relu:activations:0'dense_119/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_119/BiasAdd/ReadVariableOpReadVariableOp)dense_119_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_119/BiasAddBiasAdddense_119/MatMul:product:0(dense_119/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_119/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp!^dense_118/BiasAdd/ReadVariableOp ^dense_118/MatMul/ReadVariableOp!^dense_119/BiasAdd/ReadVariableOp ^dense_119/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2D
 dense_118/BiasAdd/ReadVariableOp dense_118/BiasAdd/ReadVariableOp2B
dense_118/MatMul/ReadVariableOpdense_118/MatMul/ReadVariableOp2D
 dense_119/BiasAdd/ReadVariableOp dense_119/BiasAdd/ReadVariableOp2B
dense_119/MatMul/ReadVariableOpdense_119/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
¥

H__inference_qfunction_39_layer_call_and_return_conditional_losses_513567

states"
dense_117_513551:


dense_117_513553:
"
dense_118_513556:

dense_118_513558:"
dense_119_513561:
dense_119_513563:
identity¢!dense_117/StatefulPartitionedCall¢!dense_118/StatefulPartitionedCall¢!dense_119/StatefulPartitionedCallô
!dense_117/StatefulPartitionedCallStatefulPartitionedCallstatesdense_117_513551dense_117_513553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_117_layer_call_and_return_conditional_losses_513444
!dense_118/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0dense_118_513556dense_118_513558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_118_layer_call_and_return_conditional_losses_513461
!dense_119/StatefulPartitionedCallStatefulPartitionedCall*dense_118/StatefulPartitionedCall:output:0dense_119_513561dense_119_513563*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_119_layer_call_and_return_conditional_losses_513477y
IdentityIdentity*dense_119/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^dense_117/StatefulPartitionedCall"^dense_118/StatefulPartitionedCall"^dense_119/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2F
!dense_118/StatefulPartitionedCall!dense_118/StatefulPartitionedCall2F
!dense_119/StatefulPartitionedCall!dense_119/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
ÿ

H__inference_qfunction_39_layer_call_and_return_conditional_losses_513695

states:
(dense_117_matmul_readvariableop_resource:

7
)dense_117_biasadd_readvariableop_resource:
:
(dense_118_matmul_readvariableop_resource:
7
)dense_118_biasadd_readvariableop_resource::
(dense_119_matmul_readvariableop_resource:7
)dense_119_biasadd_readvariableop_resource:
identity¢ dense_117/BiasAdd/ReadVariableOp¢dense_117/MatMul/ReadVariableOp¢ dense_118/BiasAdd/ReadVariableOp¢dense_118/MatMul/ReadVariableOp¢ dense_119/BiasAdd/ReadVariableOp¢dense_119/MatMul/ReadVariableOp
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0}
dense_117/MatMulMatMulstates'dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_118/MatMul/ReadVariableOpReadVariableOp(dense_118_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_118/MatMulMatMuldense_117/Relu:activations:0'dense_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_118/BiasAdd/ReadVariableOpReadVariableOp)dense_118_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_118/BiasAddBiasAdddense_118/MatMul:product:0(dense_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_118/ReluReludense_118/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_119/MatMul/ReadVariableOpReadVariableOp(dense_119_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_119/MatMulMatMuldense_118/Relu:activations:0'dense_119/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_119/BiasAdd/ReadVariableOpReadVariableOp)dense_119_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_119/BiasAddBiasAdddense_119/MatMul:product:0(dense_119/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_119/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp!^dense_118/BiasAdd/ReadVariableOp ^dense_118/MatMul/ReadVariableOp!^dense_119/BiasAdd/ReadVariableOp ^dense_119/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2D
 dense_118/BiasAdd/ReadVariableOp dense_118/BiasAdd/ReadVariableOp2B
dense_118/MatMul/ReadVariableOpdense_118/MatMul/ReadVariableOp2D
 dense_119/BiasAdd/ReadVariableOp dense_119/BiasAdd/ReadVariableOp2B
dense_119/MatMul/ReadVariableOpdense_119/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
Ä

*__inference_dense_119_layer_call_fn_513891

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_119_layer_call_and_return_conditional_losses_513477o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
 
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513637
input_1"
dense_117_513621:


dense_117_513623:
"
dense_118_513626:

dense_118_513628:"
dense_119_513631:
dense_119_513633:
identity¢!dense_117/StatefulPartitionedCall¢!dense_118/StatefulPartitionedCall¢!dense_119/StatefulPartitionedCallõ
!dense_117/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_117_513621dense_117_513623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_117_layer_call_and_return_conditional_losses_513444
!dense_118/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0dense_118_513626dense_118_513628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_118_layer_call_and_return_conditional_losses_513461
!dense_119/StatefulPartitionedCallStatefulPartitionedCall*dense_118/StatefulPartitionedCall:output:0dense_119_513631dense_119_513633*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_119_layer_call_and_return_conditional_losses_513477y
IdentityIdentity*dense_119/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^dense_117/StatefulPartitionedCall"^dense_118/StatefulPartitionedCall"^dense_119/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2F
!dense_118/StatefulPartitionedCall!dense_118/StatefulPartitionedCall2F
!dense_119/StatefulPartitionedCall!dense_119/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
¦
ú
"__inference__traced_restore_514010
file_prefix3
!assignvariableop_dense_119_kernel:/
!assignvariableop_1_dense_119_bias:5
#assignvariableop_2_dense_117_kernel:

/
!assignvariableop_3_dense_117_bias:
5
#assignvariableop_4_dense_118_kernel:
/
!assignvariableop_5_dense_118_bias:

identity_7¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*±
value§B¤B.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B Á
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_dense_119_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_119_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_117_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_117_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_118_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_118_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ö

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: Ä
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ë
ä
__inference_call_513767

states:
(dense_117_matmul_readvariableop_resource:

7
)dense_117_biasadd_readvariableop_resource:
:
(dense_118_matmul_readvariableop_resource:
7
)dense_118_biasadd_readvariableop_resource::
(dense_119_matmul_readvariableop_resource:7
)dense_119_biasadd_readvariableop_resource:
identity¢ dense_117/BiasAdd/ReadVariableOp¢dense_117/MatMul/ReadVariableOp¢ dense_118/BiasAdd/ReadVariableOp¢dense_118/MatMul/ReadVariableOp¢ dense_119/BiasAdd/ReadVariableOp¢dense_119/MatMul/ReadVariableOp
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0t
dense_117/MatMulMatMulstates'dense_117/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
[
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*
_output_shapes

:

dense_118/MatMul/ReadVariableOpReadVariableOp(dense_118_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_118/MatMulMatMuldense_117/Relu:activations:0'dense_118/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_118/BiasAdd/ReadVariableOpReadVariableOp)dense_118_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_118/BiasAddBiasAdddense_118/MatMul:product:0(dense_118/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:[
dense_118/ReluReludense_118/BiasAdd:output:0*
T0*
_output_shapes

:
dense_119/MatMul/ReadVariableOpReadVariableOp(dense_119_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_119/MatMulMatMuldense_118/Relu:activations:0'dense_119/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_119/BiasAdd/ReadVariableOpReadVariableOp)dense_119_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_119/BiasAddBiasAdddense_119/MatMul:product:0(dense_119/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_119/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp!^dense_118/BiasAdd/ReadVariableOp ^dense_118/MatMul/ReadVariableOp!^dense_119/BiasAdd/ReadVariableOp ^dense_119/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2D
 dense_118/BiasAdd/ReadVariableOp dense_118/BiasAdd/ReadVariableOp2B
dense_118/MatMul/ReadVariableOpdense_118/MatMul/ReadVariableOp2D
 dense_119/BiasAdd/ReadVariableOp dense_119/BiasAdd/ReadVariableOp2B
dense_119/MatMul/ReadVariableOpdense_119/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ
<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ýK

input_layer
hidden_layers
output_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
call

signatures"
_tf_keras_model
"
_tf_keras_input_layer
.
0
1"
trackable_list_wrapper
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
õ2ò
-__inference_qfunction_39_layer_call_fn_513499
-__inference_qfunction_39_layer_call_fn_513654
-__inference_qfunction_39_layer_call_fn_513671
-__inference_qfunction_39_layer_call_fn_513599³
ª²¦
FullArgSpec)
args!
jself
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
á2Þ
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513695
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513719
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513618
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513637³
ª²¦
FullArgSpec)
args!
jself
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÌBÉ
!__inference__wrapped_model_513426input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¾2»
__inference_call_513743
__inference_call_513767
__inference_call_513791
__inference_call_513815
__inference_call_513839
__inference_call_513863¢
²
FullArgSpec
args
jself
jstates
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
 serving_default"
signature_map
»

kernel
bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
": 2dense_119/kernel
:2dense_119/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_119_layer_call_fn_513891¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_119_layer_call_and_return_conditional_losses_513901¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 

2dense_117/kernel
:
2dense_117/bias
": 
2dense_118/kernel
:2dense_118/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ËBÈ
$__inference_signature_wrapper_513882input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_117_layer_call_fn_513910¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_117_layer_call_and_return_conditional_losses_513921¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_118_layer_call_fn_513930¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_118_layer_call_and_return_conditional_losses_513941¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
!__inference__wrapped_model_513426o0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ\
__inference_call_513743A&¢#
¢

states

ª "\
__inference_call_513767A&¢#
¢

states

ª "\
__inference_call_513791A&¢#
¢

states

ª "\
__inference_call_513815A&¢#
¢

states

ª "\
__inference_call_513839A&¢#
¢

states

ª "n
__inference_call_513863S/¢,
%¢"
 
statesÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_117_layer_call_and_return_conditional_losses_513921\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 }
*__inference_dense_117_layer_call_fn_513910O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
¥
E__inference_dense_118_layer_call_and_return_conditional_losses_513941\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_118_layer_call_fn_513930O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_119_layer_call_and_return_conditional_losses_513901\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_119_layer_call_fn_513891O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ±
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513618e4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513637e4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513695d3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
H__inference_qfunction_39_layer_call_and_return_conditional_losses_513719d3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_qfunction_39_layer_call_fn_513499X4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_qfunction_39_layer_call_fn_513599X4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_qfunction_39_layer_call_fn_513654W3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_qfunction_39_layer_call_fn_513671W3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ¢
$__inference_signature_wrapper_513882z;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ
"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ