
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
 "serve*2.8.22v2.8.2-0-g2ea19cbb5758û¥
z
dense_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_98/kernel
s
#dense_98/kernel/Read/ReadVariableOpReadVariableOpdense_98/kernel*
_output_shapes

:*
dtype0
r
dense_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_98/bias
k
!dense_98/bias/Read/ReadVariableOpReadVariableOpdense_98/bias*
_output_shapes
:*
dtype0
z
dense_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

* 
shared_namedense_96/kernel
s
#dense_96/kernel/Read/ReadVariableOpReadVariableOpdense_96/kernel*
_output_shapes

:

*
dtype0
r
dense_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_96/bias
k
!dense_96/bias/Read/ReadVariableOpReadVariableOpdense_96/bias*
_output_shapes
:
*
dtype0
z
dense_97/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_97/kernel
s
#dense_97/kernel/Read/ReadVariableOpReadVariableOpdense_97/kernel*
_output_shapes

:
*
dtype0
r
dense_97/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_97/bias
k
!dense_97/bias/Read/ReadVariableOpReadVariableOpdense_97/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ê
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¥
valueB B
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
WQ
VARIABLE_VALUEdense_98/kernel.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense_98/bias,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_96/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_96/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_97/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_97/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
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


StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_96/kerneldense_96/biasdense_97/kerneldense_97/biasdense_98/kerneldense_98/bias*
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
$__inference_signature_wrapper_505292
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ù
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_98/kernel/Read/ReadVariableOp!dense_98/bias/Read/ReadVariableOp#dense_96/kernel/Read/ReadVariableOp!dense_96/bias/Read/ReadVariableOp#dense_97/kernel/Read/ReadVariableOp!dense_97/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_505392
ü
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_98/kerneldense_98/biasdense_96/kerneldense_96/biasdense_97/kerneldense_97/bias*
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
"__inference__traced_restore_505420ø
ò

-__inference_qfunction_32_layer_call_fn_505064

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
H__inference_qfunction_32_layer_call_and_return_conditional_losses_504894o
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


H__inference_qfunction_32_layer_call_and_return_conditional_losses_504977

states!
dense_96_504961:


dense_96_504963:
!
dense_97_504966:

dense_97_504968:!
dense_98_504971:
dense_98_504973:
identity¢ dense_96/StatefulPartitionedCall¢ dense_97/StatefulPartitionedCall¢ dense_98/StatefulPartitionedCallð
 dense_96/StatefulPartitionedCallStatefulPartitionedCallstatesdense_96_504961dense_96_504963*
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
D__inference_dense_96_layer_call_and_return_conditional_losses_504854
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_504966dense_97_504968*
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
GPU 2J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_504871
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_504971dense_98_504973*
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
GPU 2J 8 *M
fHRF
D__inference_dense_98_layer_call_and_return_conditional_losses_504887x
IdentityIdentity)dense_98/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
«
Ø
__inference_call_505225

states9
'dense_96_matmul_readvariableop_resource:

6
(dense_96_biasadd_readvariableop_resource:
9
'dense_97_matmul_readvariableop_resource:
6
(dense_97_biasadd_readvariableop_resource:9
'dense_98_matmul_readvariableop_resource:6
(dense_98_biasadd_readvariableop_resource:
identity¢dense_96/BiasAdd/ReadVariableOp¢dense_96/MatMul/ReadVariableOp¢dense_97/BiasAdd/ReadVariableOp¢dense_97/MatMul/ReadVariableOp¢dense_98/BiasAdd/ReadVariableOp¢dense_98/MatMul/ReadVariableOp
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0r
dense_96/MatMulMatMulstates&dense_96/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
Y
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*
_output_shapes

:

dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense_97/ReluReludense_97/BiasAdd:output:0*
T0*
_output_shapes

:
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_98/MatMulMatMuldense_97/Relu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_98/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp ^dense_98/BiasAdd/ReadVariableOp^dense_98/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2B
dense_98/BiasAdd/ReadVariableOpdense_98/BiasAdd/ReadVariableOp2@
dense_98/MatMul/ReadVariableOpdense_98/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates
«
Ø
__inference_call_505177

states9
'dense_96_matmul_readvariableop_resource:

6
(dense_96_biasadd_readvariableop_resource:
9
'dense_97_matmul_readvariableop_resource:
6
(dense_97_biasadd_readvariableop_resource:9
'dense_98_matmul_readvariableop_resource:6
(dense_98_biasadd_readvariableop_resource:
identity¢dense_96/BiasAdd/ReadVariableOp¢dense_96/MatMul/ReadVariableOp¢dense_97/BiasAdd/ReadVariableOp¢dense_97/MatMul/ReadVariableOp¢dense_98/BiasAdd/ReadVariableOp¢dense_98/MatMul/ReadVariableOp
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0r
dense_96/MatMulMatMulstates&dense_96/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
Y
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*
_output_shapes

:

dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense_97/ReluReludense_97/BiasAdd:output:0*
T0*
_output_shapes

:
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_98/MatMulMatMuldense_97/Relu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_98/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp ^dense_98/BiasAdd/ReadVariableOp^dense_98/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2B
dense_98/BiasAdd/ReadVariableOpdense_98/BiasAdd/ReadVariableOp2@
dense_98/MatMul/ReadVariableOpdense_98/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates
¿

H__inference_qfunction_32_layer_call_and_return_conditional_losses_505129

states9
'dense_96_matmul_readvariableop_resource:

6
(dense_96_biasadd_readvariableop_resource:
9
'dense_97_matmul_readvariableop_resource:
6
(dense_97_biasadd_readvariableop_resource:9
'dense_98_matmul_readvariableop_resource:6
(dense_98_biasadd_readvariableop_resource:
identity¢dense_96/BiasAdd/ReadVariableOp¢dense_96/MatMul/ReadVariableOp¢dense_97/BiasAdd/ReadVariableOp¢dense_97/MatMul/ReadVariableOp¢dense_98/BiasAdd/ReadVariableOp¢dense_98/MatMul/ReadVariableOp
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0{
dense_96/MatMulMatMulstates&dense_96/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_97/ReluReludense_97/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_98/MatMulMatMuldense_97/Relu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_98/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp ^dense_98/BiasAdd/ReadVariableOp^dense_98/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2B
dense_98/BiasAdd/ReadVariableOpdense_98/BiasAdd/ReadVariableOp2@
dense_98/MatMul/ReadVariableOpdense_98/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
«
Ø
__inference_call_505153

states9
'dense_96_matmul_readvariableop_resource:

6
(dense_96_biasadd_readvariableop_resource:
9
'dense_97_matmul_readvariableop_resource:
6
(dense_97_biasadd_readvariableop_resource:9
'dense_98_matmul_readvariableop_resource:6
(dense_98_biasadd_readvariableop_resource:
identity¢dense_96/BiasAdd/ReadVariableOp¢dense_96/MatMul/ReadVariableOp¢dense_97/BiasAdd/ReadVariableOp¢dense_97/MatMul/ReadVariableOp¢dense_98/BiasAdd/ReadVariableOp¢dense_98/MatMul/ReadVariableOp
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0r
dense_96/MatMulMatMulstates&dense_96/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
Y
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*
_output_shapes

:

dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense_97/ReluReludense_97/BiasAdd:output:0*
T0*
_output_shapes

:
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_98/MatMulMatMuldense_97/Relu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_98/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp ^dense_98/BiasAdd/ReadVariableOp^dense_98/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2B
dense_98/BiasAdd/ReadVariableOpdense_98/BiasAdd/ReadVariableOp2@
dense_98/MatMul/ReadVariableOpdense_98/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates


õ
D__inference_dense_97_layer_call_and_return_conditional_losses_504871

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
Æ

__inference__traced_save_505392
file_prefix.
*savev2_dense_98_kernel_read_readvariableop,
(savev2_dense_98_bias_read_readvariableop.
*savev2_dense_96_kernel_read_readvariableop,
(savev2_dense_96_bias_read_readvariableop.
*savev2_dense_97_kernel_read_readvariableop,
(savev2_dense_97_bias_read_readvariableop
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
valueBB B B B B B B ¸
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_98_kernel_read_readvariableop(savev2_dense_98_bias_read_readvariableop*savev2_dense_96_kernel_read_readvariableop(savev2_dense_96_bias_read_readvariableop*savev2_dense_97_kernel_read_readvariableop(savev2_dense_97_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
«
Ø
__inference_call_505249

states9
'dense_96_matmul_readvariableop_resource:

6
(dense_96_biasadd_readvariableop_resource:
9
'dense_97_matmul_readvariableop_resource:
6
(dense_97_biasadd_readvariableop_resource:9
'dense_98_matmul_readvariableop_resource:6
(dense_98_biasadd_readvariableop_resource:
identity¢dense_96/BiasAdd/ReadVariableOp¢dense_96/MatMul/ReadVariableOp¢dense_97/BiasAdd/ReadVariableOp¢dense_97/MatMul/ReadVariableOp¢dense_98/BiasAdd/ReadVariableOp¢dense_98/MatMul/ReadVariableOp
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0r
dense_96/MatMulMatMulstates&dense_96/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
Y
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*
_output_shapes

:

dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense_97/ReluReludense_97/BiasAdd:output:0*
T0*
_output_shapes

:
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_98/MatMulMatMuldense_97/Relu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_98/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp ^dense_98/BiasAdd/ReadVariableOp^dense_98/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2B
dense_98/BiasAdd/ReadVariableOpdense_98/BiasAdd/ReadVariableOp2@
dense_98/MatMul/ReadVariableOpdense_98/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates
«
Ø
__inference_call_505201

states9
'dense_96_matmul_readvariableop_resource:

6
(dense_96_biasadd_readvariableop_resource:
9
'dense_97_matmul_readvariableop_resource:
6
(dense_97_biasadd_readvariableop_resource:9
'dense_98_matmul_readvariableop_resource:6
(dense_98_biasadd_readvariableop_resource:
identity¢dense_96/BiasAdd/ReadVariableOp¢dense_96/MatMul/ReadVariableOp¢dense_97/BiasAdd/ReadVariableOp¢dense_97/MatMul/ReadVariableOp¢dense_98/BiasAdd/ReadVariableOp¢dense_98/MatMul/ReadVariableOp
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0r
dense_96/MatMulMatMulstates&dense_96/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:

dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
Y
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*
_output_shapes

:

dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense_97/ReluReludense_97/BiasAdd:output:0*
T0*
_output_shapes

:
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_98/MatMulMatMuldense_97/Relu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_98/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp ^dense_98/BiasAdd/ReadVariableOp^dense_98/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
: : : : : : 2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2B
dense_98/BiasAdd/ReadVariableOpdense_98/BiasAdd/ReadVariableOp2@
dense_98/MatMul/ReadVariableOpdense_98/MatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_namestates

Ø
__inference_call_505273

states9
'dense_96_matmul_readvariableop_resource:

6
(dense_96_biasadd_readvariableop_resource:
9
'dense_97_matmul_readvariableop_resource:
6
(dense_97_biasadd_readvariableop_resource:9
'dense_98_matmul_readvariableop_resource:6
(dense_98_biasadd_readvariableop_resource:
identity¢dense_96/BiasAdd/ReadVariableOp¢dense_96/MatMul/ReadVariableOp¢dense_97/BiasAdd/ReadVariableOp¢dense_97/MatMul/ReadVariableOp¢dense_98/BiasAdd/ReadVariableOp¢dense_98/MatMul/ReadVariableOp
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0{
dense_96/MatMulMatMulstates&dense_96/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_97/ReluReludense_97/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_98/MatMulMatMuldense_97/Relu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_98/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp ^dense_98/BiasAdd/ReadVariableOp^dense_98/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2B
dense_98/BiasAdd/ReadVariableOpdense_98/BiasAdd/ReadVariableOp2@
dense_98/MatMul/ReadVariableOpdense_98/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
õ

-__inference_qfunction_32_layer_call_fn_505009
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
H__inference_qfunction_32_layer_call_and_return_conditional_losses_504977o
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

Ø
__inference_call_504821

states9
'dense_96_matmul_readvariableop_resource:

6
(dense_96_biasadd_readvariableop_resource:
9
'dense_97_matmul_readvariableop_resource:
6
(dense_97_biasadd_readvariableop_resource:9
'dense_98_matmul_readvariableop_resource:6
(dense_98_biasadd_readvariableop_resource:
identity¢dense_96/BiasAdd/ReadVariableOp¢dense_96/MatMul/ReadVariableOp¢dense_97/BiasAdd/ReadVariableOp¢dense_97/MatMul/ReadVariableOp¢dense_98/BiasAdd/ReadVariableOp¢dense_98/MatMul/ReadVariableOp
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0{
dense_96/MatMulMatMulstates&dense_96/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_97/ReluReludense_97/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_98/MatMulMatMuldense_97/Relu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_98/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp ^dense_98/BiasAdd/ReadVariableOp^dense_98/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2B
dense_98/BiasAdd/ReadVariableOpdense_98/BiasAdd/ReadVariableOp2@
dense_98/MatMul/ReadVariableOpdense_98/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
ò

-__inference_qfunction_32_layer_call_fn_505081

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
H__inference_qfunction_32_layer_call_and_return_conditional_losses_504977o
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


õ
D__inference_dense_97_layer_call_and_return_conditional_losses_505351

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


Æ
!__inference__wrapped_model_504836
input_1%
qfunction_32_504822:

!
qfunction_32_504824:
%
qfunction_32_504826:
!
qfunction_32_504828:%
qfunction_32_504830:!
qfunction_32_504832:
identity¢$qfunction_32/StatefulPartitionedCall¬
$qfunction_32/StatefulPartitionedCallStatefulPartitionedCallinput_1qfunction_32_504822qfunction_32_504824qfunction_32_504826qfunction_32_504828qfunction_32_504830qfunction_32_504832*
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
__inference_call_504821|
IdentityIdentity-qfunction_32/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
NoOpNoOp%^qfunction_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2L
$qfunction_32/StatefulPartitionedCall$qfunction_32/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Â

)__inference_dense_97_layer_call_fn_505340

inputs
unknown:

	unknown_0:
identity¢StatefulPartitionedCallÙ
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
GPU 2J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_504871o
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


õ
D__inference_dense_96_layer_call_and_return_conditional_losses_504854

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
Ç	
õ
D__inference_dense_98_layer_call_and_return_conditional_losses_505311

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
Â

)__inference_dense_98_layer_call_fn_505301

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÙ
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
GPU 2J 8 *M
fHRF
D__inference_dense_98_layer_call_and_return_conditional_losses_504887o
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

ô
"__inference__traced_restore_505420
file_prefix2
 assignvariableop_dense_98_kernel:.
 assignvariableop_1_dense_98_bias:4
"assignvariableop_2_dense_96_kernel:

.
 assignvariableop_3_dense_96_bias:
4
"assignvariableop_4_dense_97_kernel:
.
 assignvariableop_5_dense_97_bias:

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
:
AssignVariableOpAssignVariableOp assignvariableop_dense_98_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_98_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_96_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_96_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_97_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_97_biasIdentity_5:output:0"/device:CPU:0*
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
Ç	
õ
D__inference_dense_98_layer_call_and_return_conditional_losses_504887

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


H__inference_qfunction_32_layer_call_and_return_conditional_losses_505028
input_1!
dense_96_505012:


dense_96_505014:
!
dense_97_505017:

dense_97_505019:!
dense_98_505022:
dense_98_505024:
identity¢ dense_96/StatefulPartitionedCall¢ dense_97/StatefulPartitionedCall¢ dense_98/StatefulPartitionedCallñ
 dense_96/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_96_505012dense_96_505014*
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
D__inference_dense_96_layer_call_and_return_conditional_losses_504854
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_505017dense_97_505019*
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
GPU 2J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_504871
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_505022dense_98_505024*
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
GPU 2J 8 *M
fHRF
D__inference_dense_98_layer_call_and_return_conditional_losses_504887x
IdentityIdentity)dense_98/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1


H__inference_qfunction_32_layer_call_and_return_conditional_losses_505047
input_1!
dense_96_505031:


dense_96_505033:
!
dense_97_505036:

dense_97_505038:!
dense_98_505041:
dense_98_505043:
identity¢ dense_96/StatefulPartitionedCall¢ dense_97/StatefulPartitionedCall¢ dense_98/StatefulPartitionedCallñ
 dense_96/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_96_505031dense_96_505033*
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
D__inference_dense_96_layer_call_and_return_conditional_losses_504854
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_505036dense_97_505038*
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
GPU 2J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_504871
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_505041dense_98_505043*
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
GPU 2J 8 *M
fHRF
D__inference_dense_98_layer_call_and_return_conditional_losses_504887x
IdentityIdentity)dense_98/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
¿

H__inference_qfunction_32_layer_call_and_return_conditional_losses_505105

states9
'dense_96_matmul_readvariableop_resource:

6
(dense_96_biasadd_readvariableop_resource:
9
'dense_97_matmul_readvariableop_resource:
6
(dense_97_biasadd_readvariableop_resource:9
'dense_98_matmul_readvariableop_resource:6
(dense_98_biasadd_readvariableop_resource:
identity¢dense_96/BiasAdd/ReadVariableOp¢dense_96/MatMul/ReadVariableOp¢dense_97/BiasAdd/ReadVariableOp¢dense_97/MatMul/ReadVariableOp¢dense_98/BiasAdd/ReadVariableOp¢dense_98/MatMul/ReadVariableOp
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0{
dense_96/MatMulMatMulstates&dense_96/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_97/ReluReludense_97/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_98/MatMulMatMuldense_97/Relu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_98/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp ^dense_98/BiasAdd/ReadVariableOp^dense_98/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2B
dense_98/BiasAdd/ReadVariableOpdense_98/BiasAdd/ReadVariableOp2@
dense_98/MatMul/ReadVariableOpdense_98/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
Â

)__inference_dense_96_layer_call_fn_505320

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
D__inference_dense_96_layer_call_and_return_conditional_losses_504854o
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
Å
þ
$__inference_signature_wrapper_505292
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
!__inference__wrapped_model_504836o
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
õ

-__inference_qfunction_32_layer_call_fn_504909
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
H__inference_qfunction_32_layer_call_and_return_conditional_losses_504894o
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


H__inference_qfunction_32_layer_call_and_return_conditional_losses_504894

states!
dense_96_504855:


dense_96_504857:
!
dense_97_504872:

dense_97_504874:!
dense_98_504888:
dense_98_504890:
identity¢ dense_96/StatefulPartitionedCall¢ dense_97/StatefulPartitionedCall¢ dense_98/StatefulPartitionedCallð
 dense_96/StatefulPartitionedCallStatefulPartitionedCallstatesdense_96_504855dense_96_504857*
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
D__inference_dense_96_layer_call_and_return_conditional_losses_504854
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_504872dense_97_504874*
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
GPU 2J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_504871
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_504888dense_98_504890*
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
GPU 2J 8 *M
fHRF
D__inference_dense_98_layer_call_and_return_conditional_losses_504887x
IdentityIdentity)dense_98/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : : : : : 2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates


õ
D__inference_dense_96_layer_call_and_return_conditional_losses_505331

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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ëK
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
-__inference_qfunction_32_layer_call_fn_504909
-__inference_qfunction_32_layer_call_fn_505064
-__inference_qfunction_32_layer_call_fn_505081
-__inference_qfunction_32_layer_call_fn_505009³
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
H__inference_qfunction_32_layer_call_and_return_conditional_losses_505105
H__inference_qfunction_32_layer_call_and_return_conditional_losses_505129
H__inference_qfunction_32_layer_call_and_return_conditional_losses_505028
H__inference_qfunction_32_layer_call_and_return_conditional_losses_505047³
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
!__inference__wrapped_model_504836input_1"
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
__inference_call_505153
__inference_call_505177
__inference_call_505201
__inference_call_505225
__inference_call_505249
__inference_call_505273¢
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
!:2dense_98/kernel
:2dense_98/bias
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
Ó2Ð
)__inference_dense_98_layer_call_fn_505301¢
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
D__inference_dense_98_layer_call_and_return_conditional_losses_505311¢
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
2dense_96/kernel
:
2dense_96/bias
!:
2dense_97/kernel
:2dense_97/bias
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
$__inference_signature_wrapper_505292input_1"
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
)__inference_dense_96_layer_call_fn_505320¢
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
D__inference_dense_96_layer_call_and_return_conditional_losses_505331¢
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
Ó2Ð
)__inference_dense_97_layer_call_fn_505340¢
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
D__inference_dense_97_layer_call_and_return_conditional_losses_505351¢
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
!__inference__wrapped_model_504836o0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ\
__inference_call_505153A&¢#
¢

states

ª "\
__inference_call_505177A&¢#
¢

states

ª "\
__inference_call_505201A&¢#
¢

states

ª "\
__inference_call_505225A&¢#
¢

states

ª "\
__inference_call_505249A&¢#
¢

states

ª "n
__inference_call_505273S/¢,
%¢"
 
statesÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_96_layer_call_and_return_conditional_losses_505331\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 |
)__inference_dense_96_layer_call_fn_505320O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
¤
D__inference_dense_97_layer_call_and_return_conditional_losses_505351\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_97_layer_call_fn_505340O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_98_layer_call_and_return_conditional_losses_505311\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_98_layer_call_fn_505301O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ±
H__inference_qfunction_32_layer_call_and_return_conditional_losses_505028e4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
H__inference_qfunction_32_layer_call_and_return_conditional_losses_505047e4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
H__inference_qfunction_32_layer_call_and_return_conditional_losses_505105d3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
H__inference_qfunction_32_layer_call_and_return_conditional_losses_505129d3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_qfunction_32_layer_call_fn_504909X4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_qfunction_32_layer_call_fn_505009X4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_qfunction_32_layer_call_fn_505064W3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_qfunction_32_layer_call_fn_505081W3¢0
)¢&
 
statesÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ¢
$__inference_signature_wrapper_505292z;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ
"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ