¯
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
 "serve*2.8.22v2.8.2-0-g2ea19cbb5758ª
|
dense_101/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_101/kernel
u
$dense_101/kernel/Read/ReadVariableOpReadVariableOpdense_101/kernel*
_output_shapes

:*
dtype0
t
dense_101/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_101/bias
m
"dense_101/bias/Read/ReadVariableOpReadVariableOpdense_101/bias*
_output_shapes
:*
dtype0
z
dense_99/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

* 
shared_namedense_99/kernel
s
#dense_99/kernel/Read/ReadVariableOpReadVariableOpdense_99/kernel*
_output_shapes

:

*
dtype0
r
dense_99/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_99/bias
k
!dense_99/bias/Read/ReadVariableOpReadVariableOpdense_99/bias*
_output_shapes
:
*
dtype0
|
dense_100/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_namedense_100/kernel
u
$dense_100/kernel/Read/ReadVariableOpReadVariableOpdense_100/kernel*
_output_shapes

:
*
dtype0
t
dense_100/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_100/bias
m
"dense_100/bias/Read/ReadVariableOpReadVariableOpdense_100/bias*
_output_shapes
:*
dtype0

NoOpNoOp
î
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*©
valueB B
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
VARIABLE_VALUEdense_101/kernel.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_101/bias,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
OI
VARIABLE_VALUEdense_99/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_99/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_100/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_100/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
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

¢
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_99/kerneldense_99/biasdense_100/kerneldense_100/biasdense_101/kerneldense_101/bias*
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
$__inference_signature_wrapper_506526
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ý
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_101/kernel/Read/ReadVariableOp"dense_101/bias/Read/ReadVariableOp#dense_99/kernel/Read/ReadVariableOp!dense_99/bias/Read/ReadVariableOp$dense_100/kernel/Read/ReadVariableOp"dense_100/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_506626

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_101/kerneldense_101/biasdense_99/kerneldense_99/biasdense_100/kerneldense_100/bias*
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
"__inference__traced_restore_506654ü
Å
þ
$__inference_signature_wrapper_506526
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
!__inference__wrapped_model_506070o
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


Æ
!__inference__wrapped_model_506070
input_1%
qfunction_33_506056:

!
qfunction_33_506058:
%
qfunction_33_506060:
!
qfunction_33_506062:%
qfunction_33_506064:!
qfunction_33_506066:
identity¢$qfunction_33/StatefulPartitionedCall¬
$qfunction_33/StatefulPartitionedCallStatefulPartitionedCallinput_1qfunction_33_506056qfunction_33_506058qfunction_33_506060qfunction_33_506062qfunction_33_506064qfunction_33_506066*
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
__inference_call_506055|
IdentityIdentity-qfunction_33/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
NoOpNoOp%^qfunction_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2L
$qfunction_33/StatefulPartitionedCall$qfunction_33/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Ä

*__inference_dense_100_layer_call_fn_506574

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
E__inference_dense_100_layer_call_and_return_conditional_losses_506105o
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
Õ
à
__inference_call_506435

states9
'dense_99_matmul_readvariableop_resource:

6
(dense_99_biasadd_readvariableop_resource:
:
(dense_100_matmul_readvariableop_resource:
7
)dense_100_biasadd_readvariableop_resource::
(dense_101_matmul_readvariableop_resource:7
)dense_101_biasadd_readvariableop_resource:
identity¢ dense_100/BiasAdd/ReadVariableOp¢dense_100/MatMul/ReadVariableOp¢ dense_101/BiasAdd/ReadVariableOp¢dense_101/MatMul/ReadVariableOp¢dense_99/BiasAdd/ReadVariableOp¢dense_99/MatMul/ReadVariableOp
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0r
dense_99/MatMulMatMulstates&dense_99/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
Y
dense_99/ReluReludense_99/BiasAdd:output:0*
T0*
_output_shapes

:

dense_100/MatMul/ReadVariableOpReadVariableOp(dense_100_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_100/MatMulMatMuldense_99/Relu:activations:0'dense_100/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_100/BiasAddBiasAdddense_100/MatMul:product:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:[
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*
_output_shapes

:
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_101/MatMulMatMuldense_100/Relu:activations:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_101/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_100/BiasAdd/ReadVariableOp ^dense_100/MatMul/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2D
 dense_100/BiasAdd/ReadVariableOp dense_100/BiasAdd/ReadVariableOp2B
dense_100/MatMul/ReadVariableOpdense_100/MatMul/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates


H__inference_qfunction_33_layer_call_and_return_conditional_losses_506128

states!
dense_99_506089:


dense_99_506091:
"
dense_100_506106:

dense_100_506108:"
dense_101_506122:
dense_101_506124:
identity¢!dense_100/StatefulPartitionedCall¢!dense_101/StatefulPartitionedCall¢ dense_99/StatefulPartitionedCallð
 dense_99/StatefulPartitionedCallStatefulPartitionedCallstatesdense_99_506089dense_99_506091*
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
GPU 2J 8 *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_506088
!dense_100/StatefulPartitionedCallStatefulPartitionedCall)dense_99/StatefulPartitionedCall:output:0dense_100_506106dense_100_506108*
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
E__inference_dense_100_layer_call_and_return_conditional_losses_506105
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_506122dense_101_506124*
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
E__inference_dense_101_layer_call_and_return_conditional_losses_506121y
IdentityIdentity*dense_101/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates


H__inference_qfunction_33_layer_call_and_return_conditional_losses_506281
input_1!
dense_99_506265:


dense_99_506267:
"
dense_100_506270:

dense_100_506272:"
dense_101_506275:
dense_101_506277:
identity¢!dense_100/StatefulPartitionedCall¢!dense_101/StatefulPartitionedCall¢ dense_99/StatefulPartitionedCallñ
 dense_99/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_99_506265dense_99_506267*
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
GPU 2J 8 *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_506088
!dense_100/StatefulPartitionedCallStatefulPartitionedCall)dense_99/StatefulPartitionedCall:output:0dense_100_506270dense_100_506272*
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
E__inference_dense_100_layer_call_and_return_conditional_losses_506105
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_506275dense_101_506277*
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
E__inference_dense_101_layer_call_and_return_conditional_losses_506121y
IdentityIdentity*dense_101/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Õ
à
__inference_call_506387

states9
'dense_99_matmul_readvariableop_resource:

6
(dense_99_biasadd_readvariableop_resource:
:
(dense_100_matmul_readvariableop_resource:
7
)dense_100_biasadd_readvariableop_resource::
(dense_101_matmul_readvariableop_resource:7
)dense_101_biasadd_readvariableop_resource:
identity¢ dense_100/BiasAdd/ReadVariableOp¢dense_100/MatMul/ReadVariableOp¢ dense_101/BiasAdd/ReadVariableOp¢dense_101/MatMul/ReadVariableOp¢dense_99/BiasAdd/ReadVariableOp¢dense_99/MatMul/ReadVariableOp
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0r
dense_99/MatMulMatMulstates&dense_99/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
Y
dense_99/ReluReludense_99/BiasAdd:output:0*
T0*
_output_shapes

:

dense_100/MatMul/ReadVariableOpReadVariableOp(dense_100_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_100/MatMulMatMuldense_99/Relu:activations:0'dense_100/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_100/BiasAddBiasAdddense_100/MatMul:product:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:[
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*
_output_shapes

:
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_101/MatMulMatMuldense_100/Relu:activations:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_101/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_100/BiasAdd/ReadVariableOp ^dense_100/MatMul/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2D
 dense_100/BiasAdd/ReadVariableOp dense_100/BiasAdd/ReadVariableOp2B
dense_100/MatMul/ReadVariableOpdense_100/MatMul/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates
È	
ö
E__inference_dense_101_layer_call_and_return_conditional_losses_506545

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


H__inference_qfunction_33_layer_call_and_return_conditional_losses_506262
input_1!
dense_99_506246:


dense_99_506248:
"
dense_100_506251:

dense_100_506253:"
dense_101_506256:
dense_101_506258:
identity¢!dense_100/StatefulPartitionedCall¢!dense_101/StatefulPartitionedCall¢ dense_99/StatefulPartitionedCallñ
 dense_99/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_99_506246dense_99_506248*
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
GPU 2J 8 *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_506088
!dense_100/StatefulPartitionedCallStatefulPartitionedCall)dense_99/StatefulPartitionedCall:output:0dense_100_506251dense_100_506253*
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
E__inference_dense_100_layer_call_and_return_conditional_losses_506105
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_506256dense_101_506258*
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
E__inference_dense_101_layer_call_and_return_conditional_losses_506121y
IdentityIdentity*dense_101/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Õ
à
__inference_call_506459

states9
'dense_99_matmul_readvariableop_resource:

6
(dense_99_biasadd_readvariableop_resource:
:
(dense_100_matmul_readvariableop_resource:
7
)dense_100_biasadd_readvariableop_resource::
(dense_101_matmul_readvariableop_resource:7
)dense_101_biasadd_readvariableop_resource:
identity¢ dense_100/BiasAdd/ReadVariableOp¢dense_100/MatMul/ReadVariableOp¢ dense_101/BiasAdd/ReadVariableOp¢dense_101/MatMul/ReadVariableOp¢dense_99/BiasAdd/ReadVariableOp¢dense_99/MatMul/ReadVariableOp
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0r
dense_99/MatMulMatMulstates&dense_99/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
Y
dense_99/ReluReludense_99/BiasAdd:output:0*
T0*
_output_shapes

:

dense_100/MatMul/ReadVariableOpReadVariableOp(dense_100_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_100/MatMulMatMuldense_99/Relu:activations:0'dense_100/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_100/BiasAddBiasAdddense_100/MatMul:product:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:[
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*
_output_shapes

:
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_101/MatMulMatMuldense_100/Relu:activations:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_101/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_100/BiasAdd/ReadVariableOp ^dense_100/MatMul/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2D
 dense_100/BiasAdd/ReadVariableOp dense_100/BiasAdd/ReadVariableOp2B
dense_100/MatMul/ReadVariableOpdense_100/MatMul/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates


õ
D__inference_dense_99_layer_call_and_return_conditional_losses_506088

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
õ

-__inference_qfunction_33_layer_call_fn_506243
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
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506211o
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


ö
E__inference_dense_100_layer_call_and_return_conditional_losses_506585

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
¸
à
__inference_call_506055

states9
'dense_99_matmul_readvariableop_resource:

6
(dense_99_biasadd_readvariableop_resource:
:
(dense_100_matmul_readvariableop_resource:
7
)dense_100_biasadd_readvariableop_resource::
(dense_101_matmul_readvariableop_resource:7
)dense_101_biasadd_readvariableop_resource:
identity¢ dense_100/BiasAdd/ReadVariableOp¢dense_100/MatMul/ReadVariableOp¢ dense_101/BiasAdd/ReadVariableOp¢dense_101/MatMul/ReadVariableOp¢dense_99/BiasAdd/ReadVariableOp¢dense_99/MatMul/ReadVariableOp
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0{
dense_99/MatMulMatMulstates&dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
dense_99/ReluReludense_99/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_100/MatMul/ReadVariableOpReadVariableOp(dense_100_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_100/MatMulMatMuldense_99/Relu:activations:0'dense_100/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_100/BiasAddBiasAdddense_100/MatMul:product:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_101/MatMulMatMuldense_100/Relu:activations:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_101/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_100/BiasAdd/ReadVariableOp ^dense_100/MatMul/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_100/BiasAdd/ReadVariableOp dense_100/BiasAdd/ReadVariableOp2B
dense_100/MatMul/ReadVariableOpdense_100/MatMul/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
é

H__inference_qfunction_33_layer_call_and_return_conditional_losses_506363

states9
'dense_99_matmul_readvariableop_resource:

6
(dense_99_biasadd_readvariableop_resource:
:
(dense_100_matmul_readvariableop_resource:
7
)dense_100_biasadd_readvariableop_resource::
(dense_101_matmul_readvariableop_resource:7
)dense_101_biasadd_readvariableop_resource:
identity¢ dense_100/BiasAdd/ReadVariableOp¢dense_100/MatMul/ReadVariableOp¢ dense_101/BiasAdd/ReadVariableOp¢dense_101/MatMul/ReadVariableOp¢dense_99/BiasAdd/ReadVariableOp¢dense_99/MatMul/ReadVariableOp
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0{
dense_99/MatMulMatMulstates&dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
dense_99/ReluReludense_99/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_100/MatMul/ReadVariableOpReadVariableOp(dense_100_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_100/MatMulMatMuldense_99/Relu:activations:0'dense_100/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_100/BiasAddBiasAdddense_100/MatMul:product:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_101/MatMulMatMuldense_100/Relu:activations:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_101/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_100/BiasAdd/ReadVariableOp ^dense_100/MatMul/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_100/BiasAdd/ReadVariableOp dense_100/BiasAdd/ReadVariableOp2B
dense_100/MatMul/ReadVariableOpdense_100/MatMul/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
È	
ö
E__inference_dense_101_layer_call_and_return_conditional_losses_506121

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
Õ
à
__inference_call_506411

states9
'dense_99_matmul_readvariableop_resource:

6
(dense_99_biasadd_readvariableop_resource:
:
(dense_100_matmul_readvariableop_resource:
7
)dense_100_biasadd_readvariableop_resource::
(dense_101_matmul_readvariableop_resource:7
)dense_101_biasadd_readvariableop_resource:
identity¢ dense_100/BiasAdd/ReadVariableOp¢dense_100/MatMul/ReadVariableOp¢ dense_101/BiasAdd/ReadVariableOp¢dense_101/MatMul/ReadVariableOp¢dense_99/BiasAdd/ReadVariableOp¢dense_99/MatMul/ReadVariableOp
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0r
dense_99/MatMulMatMulstates&dense_99/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
Y
dense_99/ReluReludense_99/BiasAdd:output:0*
T0*
_output_shapes

:

dense_100/MatMul/ReadVariableOpReadVariableOp(dense_100_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_100/MatMulMatMuldense_99/Relu:activations:0'dense_100/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_100/BiasAddBiasAdddense_100/MatMul:product:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:[
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*
_output_shapes

:
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_101/MatMulMatMuldense_100/Relu:activations:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_101/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_100/BiasAdd/ReadVariableOp ^dense_100/MatMul/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2D
 dense_100/BiasAdd/ReadVariableOp dense_100/BiasAdd/ReadVariableOp2B
dense_100/MatMul/ReadVariableOpdense_100/MatMul/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates
Â

)__inference_dense_99_layer_call_fn_506554

inputs
unknown:


	unknown_0:

identity¢StatefulPartitionedCallÙ
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
GPU 2J 8 *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_506088o
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


õ
D__inference_dense_99_layer_call_and_return_conditional_losses_506565

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
Õ
à
__inference_call_506483

states9
'dense_99_matmul_readvariableop_resource:

6
(dense_99_biasadd_readvariableop_resource:
:
(dense_100_matmul_readvariableop_resource:
7
)dense_100_biasadd_readvariableop_resource::
(dense_101_matmul_readvariableop_resource:7
)dense_101_biasadd_readvariableop_resource:
identity¢ dense_100/BiasAdd/ReadVariableOp¢dense_100/MatMul/ReadVariableOp¢ dense_101/BiasAdd/ReadVariableOp¢dense_101/MatMul/ReadVariableOp¢dense_99/BiasAdd/ReadVariableOp¢dense_99/MatMul/ReadVariableOp
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0r
dense_99/MatMulMatMulstates&dense_99/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
Y
dense_99/ReluReludense_99/BiasAdd:output:0*
T0*
_output_shapes

:

dense_100/MatMul/ReadVariableOpReadVariableOp(dense_100_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_100/MatMulMatMuldense_99/Relu:activations:0'dense_100/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_100/BiasAddBiasAdddense_100/MatMul:product:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:[
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*
_output_shapes

:
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_101/MatMulMatMuldense_100/Relu:activations:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:`
IdentityIdentitydense_101/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp!^dense_100/BiasAdd/ReadVariableOp ^dense_100/MatMul/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2D
 dense_100/BiasAdd/ReadVariableOp dense_100/BiasAdd/ReadVariableOp2B
dense_100/MatMul/ReadVariableOpdense_100/MatMul/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates
¸
à
__inference_call_506507

states9
'dense_99_matmul_readvariableop_resource:

6
(dense_99_biasadd_readvariableop_resource:
:
(dense_100_matmul_readvariableop_resource:
7
)dense_100_biasadd_readvariableop_resource::
(dense_101_matmul_readvariableop_resource:7
)dense_101_biasadd_readvariableop_resource:
identity¢ dense_100/BiasAdd/ReadVariableOp¢dense_100/MatMul/ReadVariableOp¢ dense_101/BiasAdd/ReadVariableOp¢dense_101/MatMul/ReadVariableOp¢dense_99/BiasAdd/ReadVariableOp¢dense_99/MatMul/ReadVariableOp
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0{
dense_99/MatMulMatMulstates&dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
dense_99/ReluReludense_99/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_100/MatMul/ReadVariableOpReadVariableOp(dense_100_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_100/MatMulMatMuldense_99/Relu:activations:0'dense_100/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_100/BiasAddBiasAdddense_100/MatMul:product:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_101/MatMulMatMuldense_100/Relu:activations:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_101/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_100/BiasAdd/ReadVariableOp ^dense_100/MatMul/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_100/BiasAdd/ReadVariableOp dense_100/BiasAdd/ReadVariableOp2B
dense_100/MatMul/ReadVariableOpdense_100/MatMul/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates


H__inference_qfunction_33_layer_call_and_return_conditional_losses_506211

states!
dense_99_506195:


dense_99_506197:
"
dense_100_506200:

dense_100_506202:"
dense_101_506205:
dense_101_506207:
identity¢!dense_100/StatefulPartitionedCall¢!dense_101/StatefulPartitionedCall¢ dense_99/StatefulPartitionedCallð
 dense_99/StatefulPartitionedCallStatefulPartitionedCallstatesdense_99_506195dense_99_506197*
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
GPU 2J 8 *M
fHRF
D__inference_dense_99_layer_call_and_return_conditional_losses_506088
!dense_100/StatefulPartitionedCallStatefulPartitionedCall)dense_99/StatefulPartitionedCall:output:0dense_100_506200dense_100_506202*
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
E__inference_dense_100_layer_call_and_return_conditional_losses_506105
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_506205dense_101_506207*
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
E__inference_dense_101_layer_call_and_return_conditional_losses_506121y
IdentityIdentity*dense_101/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
¢
ø
"__inference__traced_restore_506654
file_prefix3
!assignvariableop_dense_101_kernel:/
!assignvariableop_1_dense_101_bias:4
"assignvariableop_2_dense_99_kernel:

.
 assignvariableop_3_dense_99_bias:
5
#assignvariableop_4_dense_100_kernel:
/
!assignvariableop_5_dense_100_bias:

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
AssignVariableOpAssignVariableOp!assignvariableop_dense_101_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_101_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_99_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_99_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_100_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_100_biasIdentity_5:output:0"/device:CPU:0*
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
é

H__inference_qfunction_33_layer_call_and_return_conditional_losses_506339

states9
'dense_99_matmul_readvariableop_resource:

6
(dense_99_biasadd_readvariableop_resource:
:
(dense_100_matmul_readvariableop_resource:
7
)dense_100_biasadd_readvariableop_resource::
(dense_101_matmul_readvariableop_resource:7
)dense_101_biasadd_readvariableop_resource:
identity¢ dense_100/BiasAdd/ReadVariableOp¢dense_100/MatMul/ReadVariableOp¢ dense_101/BiasAdd/ReadVariableOp¢dense_101/MatMul/ReadVariableOp¢dense_99/BiasAdd/ReadVariableOp¢dense_99/MatMul/ReadVariableOp
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0{
dense_99/MatMulMatMulstates&dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
dense_99/ReluReludense_99/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_100/MatMul/ReadVariableOpReadVariableOp(dense_100_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_100/MatMulMatMuldense_99/Relu:activations:0'dense_100/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_100/BiasAddBiasAdddense_100/MatMul:product:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_101/MatMul/ReadVariableOpReadVariableOp(dense_101_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_101/MatMulMatMuldense_100/Relu:activations:0'dense_101/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_101/BiasAddBiasAdddense_101/MatMul:product:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_101/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_100/BiasAdd/ReadVariableOp ^dense_100/MatMul/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp ^dense_101/MatMul/ReadVariableOp ^dense_99/BiasAdd/ReadVariableOp^dense_99/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_100/BiasAdd/ReadVariableOp dense_100/BiasAdd/ReadVariableOp2B
dense_100/MatMul/ReadVariableOpdense_100/MatMul/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2B
dense_101/MatMul/ReadVariableOpdense_101/MatMul/ReadVariableOp2B
dense_99/BiasAdd/ReadVariableOpdense_99/BiasAdd/ReadVariableOp2@
dense_99/MatMul/ReadVariableOpdense_99/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
Î

__inference__traced_save_506626
file_prefix/
+savev2_dense_101_kernel_read_readvariableop-
)savev2_dense_101_bias_read_readvariableop.
*savev2_dense_99_kernel_read_readvariableop,
(savev2_dense_99_bias_read_readvariableop/
+savev2_dense_100_kernel_read_readvariableop-
)savev2_dense_100_bias_read_readvariableop
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
valueBB B B B B B B ¼
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_101_kernel_read_readvariableop)savev2_dense_101_bias_read_readvariableop*savev2_dense_99_kernel_read_readvariableop(savev2_dense_99_bias_read_readvariableop+savev2_dense_100_kernel_read_readvariableop)savev2_dense_100_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
ò

-__inference_qfunction_33_layer_call_fn_506298

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
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506128o
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
E__inference_dense_100_layer_call_and_return_conditional_losses_506105

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
ò

-__inference_qfunction_33_layer_call_fn_506315

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
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506211o
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
õ

-__inference_qfunction_33_layer_call_fn_506143
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
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506128o
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
Ä

*__inference_dense_101_layer_call_fn_506535

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
E__inference_dense_101_layer_call_and_return_conditional_losses_506121o
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
 
_user_specified_nameinputs"ÛL
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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:÷K
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
-__inference_qfunction_33_layer_call_fn_506143
-__inference_qfunction_33_layer_call_fn_506298
-__inference_qfunction_33_layer_call_fn_506315
-__inference_qfunction_33_layer_call_fn_506243³
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
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506339
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506363
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506262
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506281³
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
!__inference__wrapped_model_506070input_1"
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
__inference_call_506387
__inference_call_506411
__inference_call_506435
__inference_call_506459
__inference_call_506483
__inference_call_506507¢
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
": 2dense_101/kernel
:2dense_101/bias
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
*__inference_dense_101_layer_call_fn_506535¢
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
E__inference_dense_101_layer_call_and_return_conditional_losses_506545¢
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
!:

2dense_99/kernel
:
2dense_99/bias
": 
2dense_100/kernel
:2dense_100/bias
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
$__inference_signature_wrapper_506526input_1"
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
Ó2Ð
)__inference_dense_99_layer_call_fn_506554¢
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
î2ë
D__inference_dense_99_layer_call_and_return_conditional_losses_506565¢
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
*__inference_dense_100_layer_call_fn_506574¢
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
E__inference_dense_100_layer_call_and_return_conditional_losses_506585¢
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
!__inference__wrapped_model_506070o0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ\
__inference_call_506387A&¢#
¢

states

ª "\
__inference_call_506411A&¢#
¢

states

ª "\
__inference_call_506435A&¢#
¢

states

ª "\
__inference_call_506459A&¢#
¢

states

ª "\
__inference_call_506483A&¢#
¢

states

ª "n
__inference_call_506507S/¢,
%¢"
 
statesÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_100_layer_call_and_return_conditional_losses_506585\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_100_layer_call_fn_506574O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_101_layer_call_and_return_conditional_losses_506545\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_101_layer_call_fn_506535O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_99_layer_call_and_return_conditional_losses_506565\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 |
)__inference_dense_99_layer_call_fn_506554O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
±
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506262e4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506281e4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506339d3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
H__inference_qfunction_33_layer_call_and_return_conditional_losses_506363d3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_qfunction_33_layer_call_fn_506143X4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_qfunction_33_layer_call_fn_506243X4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_qfunction_33_layer_call_fn_506298W3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_qfunction_33_layer_call_fn_506315W3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ¢
$__inference_signature_wrapper_506526z;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ
"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ