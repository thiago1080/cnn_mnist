��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.17.02v2.17.0-rc1-2-gad6d8cc177d8��
z
bias3VarHandleOp*
_output_shapes
: *

debug_namebias3/*
dtype0*
shape:
*
shared_namebias3
[
bias3/Read/ReadVariableOpReadVariableOpbias3*
_output_shapes
:
*
dtype0
{
bias2VarHandleOp*
_output_shapes
: *

debug_namebias2/*
dtype0*
shape:�*
shared_namebias2
\
bias2/Read/ReadVariableOpReadVariableOpbias2*
_output_shapes	
:�*
dtype0
{
bias1VarHandleOp*
_output_shapes
: *

debug_namebias1/*
dtype0*
shape:�*
shared_namebias1
\
bias1/Read/ReadVariableOpReadVariableOpbias1*
_output_shapes	
:�*
dtype0
y
fc3VarHandleOp*
_output_shapes
: *

debug_namefc3/*
dtype0*
shape:	�
*
shared_namefc3
\
fc3/Read/ReadVariableOpReadVariableOpfc3*
_output_shapes
:	�
*
dtype0
z
fc2VarHandleOp*
_output_shapes
: *

debug_namefc2/*
dtype0*
shape:
��*
shared_namefc2
]
fc2/Read/ReadVariableOpReadVariableOpfc2* 
_output_shapes
:
��*
dtype0
{
fc1VarHandleOp*
_output_shapes
: *

debug_namefc1/*
dtype0*
shape:���*
shared_namefc1
^
fc1/Read/ReadVariableOpReadVariableOpfc1*!
_output_shapes
:���*
dtype0
�
conv3_2VarHandleOp*
_output_shapes
: *

debug_name
conv3_2/*
dtype0*
shape: �*
shared_name	conv3_2
l
conv3_2/Read/ReadVariableOpReadVariableOpconv3_2*'
_output_shapes
: �*
dtype0
�
conv3_1VarHandleOp*
_output_shapes
: *

debug_name
conv3_1/*
dtype0*
shape: �*
shared_name	conv3_1
l
conv3_1/Read/ReadVariableOpReadVariableOpconv3_1*'
_output_shapes
: �*
dtype0
�
conv2_2VarHandleOp*
_output_shapes
: *

debug_name
conv2_2/*
dtype0*
shape:  *
shared_name	conv2_2
k
conv2_2/Read/ReadVariableOpReadVariableOpconv2_2*&
_output_shapes
:  *
dtype0
�
conv2_1VarHandleOp*
_output_shapes
: *

debug_name
conv2_1/*
dtype0*
shape:  *
shared_name	conv2_1
k
conv2_1/Read/ReadVariableOpReadVariableOpconv2_1*&
_output_shapes
:  *
dtype0
�
conv1VarHandleOp*
_output_shapes
: *

debug_nameconv1/*
dtype0*
shape: *
shared_nameconv1
g
conv1/Read/ReadVariableOpReadVariableOpconv1*&
_output_shapes
: *
dtype0
G
serving_default_xPlaceholder*
_output_shapes
:*
dtype0
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_xconv1conv2_1conv2_2conv3_1conv3_2fc1bias1fc2bias2fc3bias3*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_2225045

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	conv1
conv2_1
conv2_2
conv3_1
conv3_2
fc1
fc2
fc3
		bias1
	
bias2
	bias3
__call__

signatures*
?9
VARIABLE_VALUEconv1 conv1/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEconv2_1"conv2_1/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEconv2_2"conv2_2/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEconv3_1"conv3_1/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEconv3_2"conv3_2/.ATTRIBUTES/VARIABLE_VALUE*
;5
VARIABLE_VALUEfc1fc1/.ATTRIBUTES/VARIABLE_VALUE*
;5
VARIABLE_VALUEfc2fc2/.ATTRIBUTES/VARIABLE_VALUE*
;5
VARIABLE_VALUEfc3fc3/.ATTRIBUTES/VARIABLE_VALUE*
?9
VARIABLE_VALUEbias1 bias1/.ATTRIBUTES/VARIABLE_VALUE*
?9
VARIABLE_VALUEbias2 bias2/.ATTRIBUTES/VARIABLE_VALUE*
?9
VARIABLE_VALUEbias3 bias3/.ATTRIBUTES/VARIABLE_VALUE*
6
trace_0
trace_1
trace_2
trace_3* 

serving_default* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv1conv2_1conv2_2conv3_1conv3_2fc1fc2fc3bias1bias2bias3Const*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_2225184
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1conv2_1conv2_2conv3_1conv3_2fc1fc2fc3bias1bias2bias3*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_2225226��
�4
�
__inference___call___2224435
x8
conv2d_readvariableop_resource: :
 conv2d_1_readvariableop_resource:  :
 conv2d_2_readvariableop_resource:  ;
 conv2d_3_readvariableop_resource: �;
 conv2d_4_readvariableop_resource: �3
matmul_readvariableop_resource:���*
add_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��,
add_1_readvariableop_resource:	�3
 matmul_2_readvariableop_resource:	�
+
add_2_readvariableop_resource:

identity��Conv2D/ReadVariableOp�Conv2D_1/ReadVariableOp�Conv2D_2/ReadVariableOp�Conv2D_3/ReadVariableOp�Conv2D_4/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�MatMul_2/ReadVariableOp�add/ReadVariableOp�add_1/ReadVariableOp�add_2/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DxConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:�N *
paddingSAME*
strides
O
ReluReluConv2D:output:0*
T0*'
_output_shapes
:�N �
	MaxPool2dMaxPoolRelu:activations:0*'
_output_shapes
:�N *
ksize
*
paddingSAME*
strides
�
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2D_1Conv2DMaxPool2d:output:0Conv2D_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:�N *
paddingSAME*
strides
S
Relu_1ReluConv2D_1:output:0*
T0*'
_output_shapes
:�N �
MaxPool2d_1MaxPoolRelu_1:activations:0*'
_output_shapes
:�N *
ksize
*
paddingSAME*
strides
�
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2D_2Conv2DMaxPool2d:output:0Conv2D_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:�N *
paddingSAME*
strides
S
Relu_2ReluConv2D_2:output:0*
T0*'
_output_shapes
:�N �
MaxPool2d_2MaxPoolRelu_2:activations:0*'
_output_shapes
:�N *
ksize
*
paddingSAME*
strides
�
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*'
_output_shapes
: �*
dtype0�
Conv2D_3Conv2DMaxPool2d_1:output:0Conv2D_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:�N�*
paddingSAME*
strides
T
Relu_3ReluConv2D_3:output:0*
T0*(
_output_shapes
:�N��
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*'
_output_shapes
: �*
dtype0�
Conv2D_4Conv2DMaxPool2d_2:output:0Conv2D_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:�N�*
paddingSAME*
strides
T
Relu_4ReluConv2D_4:output:0*
T0*(
_output_shapes
:�N�M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2Relu_3:activations:0Relu_4:activations:0concat/axis:output:0*
N*
T0*(
_output_shapes
:�N�^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� b  g
ReshapeReshapeconcat:output:0Reshape/shape:output:0*
T0*!
_output_shapes
:�N��w
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0l
MatMulMatMulReshape:output:0MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�N�k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0e
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�N�B
Relu_5Reluadd:z:0*
T0* 
_output_shapes
:
�N�z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0t
MatMul_1MatMulRelu_5:activations:0MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�N�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�N�D
Relu_6Relu	add_1:z:0*
T0* 
_output_shapes
:
�N�y
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	�
*
dtype0s
MatMul_2MatMulRelu_6:activations:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�N
n
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:
*
dtype0j
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�N
P
IdentityIdentity	add_2:z:0^NoOp*
T0*
_output_shapes
:	�N
�
NoOpNoOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^Conv2D_3/ReadVariableOp^Conv2D_4/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�N: : : : : : : : : : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp22
Conv2D_3/ReadVariableOpConv2D_3/ReadVariableOp22
Conv2D_4/ReadVariableOpConv2D_4/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp:J F
'
_output_shapes
:�N

_user_specified_namex:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
%__inference_signature_wrapper_2225045
x!
unknown: #
	unknown_0:  #
	unknown_1:  $
	unknown_2: �$
	unknown_3: �
	unknown_4:���
	unknown_5:	�
	unknown_6:
��
	unknown_7:	�
	unknown_8:	�

	unknown_9:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *%
f R
__inference___call___2225017o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:; 7

_output_shapes
:

_user_specified_namex:'#
!
_user_specified_name	2225021:'#
!
_user_specified_name	2225023:'#
!
_user_specified_name	2225025:'#
!
_user_specified_name	2225027:'#
!
_user_specified_name	2225029:'#
!
_user_specified_name	2225031:'#
!
_user_specified_name	2225033:'#
!
_user_specified_name	2225035:'	#
!
_user_specified_name	2225037:'
#
!
_user_specified_name	2225039:'#
!
_user_specified_name	2225041
�4
�
__inference___call___2224384
x8
conv2d_readvariableop_resource: :
 conv2d_1_readvariableop_resource:  :
 conv2d_2_readvariableop_resource:  ;
 conv2d_3_readvariableop_resource: �;
 conv2d_4_readvariableop_resource: �3
matmul_readvariableop_resource:���*
add_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��,
add_1_readvariableop_resource:	�3
 matmul_2_readvariableop_resource:	�
+
add_2_readvariableop_resource:

identity��Conv2D/ReadVariableOp�Conv2D_1/ReadVariableOp�Conv2D_2/ReadVariableOp�Conv2D_3/ReadVariableOp�Conv2D_4/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�MatMul_2/ReadVariableOp�add/ReadVariableOp�add_1/ReadVariableOp�add_2/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DxConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:` *
paddingSAME*
strides
N
ReluReluConv2D:output:0*
T0*&
_output_shapes
:` �
	MaxPool2dMaxPoolRelu:activations:0*&
_output_shapes
:` *
ksize
*
paddingSAME*
strides
�
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2D_1Conv2DMaxPool2d:output:0Conv2D_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:` *
paddingSAME*
strides
R
Relu_1ReluConv2D_1:output:0*
T0*&
_output_shapes
:` �
MaxPool2d_1MaxPoolRelu_1:activations:0*&
_output_shapes
:` *
ksize
*
paddingSAME*
strides
�
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2D_2Conv2DMaxPool2d:output:0Conv2D_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:` *
paddingSAME*
strides
R
Relu_2ReluConv2D_2:output:0*
T0*&
_output_shapes
:` �
MaxPool2d_2MaxPoolRelu_2:activations:0*&
_output_shapes
:` *
ksize
*
paddingSAME*
strides
�
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*'
_output_shapes
: �*
dtype0�
Conv2D_3Conv2DMaxPool2d_1:output:0Conv2D_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:`�*
paddingSAME*
strides
S
Relu_3ReluConv2D_3:output:0*
T0*'
_output_shapes
:`��
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*'
_output_shapes
: �*
dtype0�
Conv2D_4Conv2DMaxPool2d_2:output:0Conv2D_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:`�*
paddingSAME*
strides
S
Relu_4ReluConv2D_4:output:0*
T0*'
_output_shapes
:`�M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2Relu_3:activations:0Relu_4:activations:0concat/axis:output:0*
N*
T0*'
_output_shapes
:`�^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� b  f
ReshapeReshapeconcat:output:0Reshape/shape:output:0*
T0* 
_output_shapes
:
`��w
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0k
MatMulMatMulReshape:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	`�k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0d
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*
_output_shapes
:	`�A
Relu_5Reluadd:z:0*
T0*
_output_shapes
:	`�z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0s
MatMul_1MatMulRelu_5:activations:0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	`�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0j
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	`�C
Relu_6Relu	add_1:z:0*
T0*
_output_shapes
:	`�y
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	�
*
dtype0r
MatMul_2MatMulRelu_6:activations:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`
n
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:
*
dtype0i
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`
O
IdentityIdentity	add_2:z:0^NoOp*
T0*
_output_shapes

:`
�
NoOpNoOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^Conv2D_3/ReadVariableOp^Conv2D_4/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:`: : : : : : : : : : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp22
Conv2D_3/ReadVariableOpConv2D_3/ReadVariableOp22
Conv2D_4/ReadVariableOpConv2D_4/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp:I E
&
_output_shapes
:`

_user_specified_namex:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�W
�	
 __inference__traced_save_2225184
file_prefix6
read_disablecopyonread_conv1: :
 read_1_disablecopyonread_conv2_1:  :
 read_2_disablecopyonread_conv2_2:  ;
 read_3_disablecopyonread_conv3_1: �;
 read_4_disablecopyonread_conv3_2: �1
read_5_disablecopyonread_fc1:���0
read_6_disablecopyonread_fc2:
��/
read_7_disablecopyonread_fc3:	�
-
read_8_disablecopyonread_bias1:	�-
read_9_disablecopyonread_bias2:	�-
read_10_disablecopyonread_bias3:

savev2_const
identity_23��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: _
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_conv1*
_output_shapes
 �
Read/ReadVariableOpReadVariableOpread_disablecopyonread_conv1^Read/DisableCopyOnRead*&
_output_shapes
: *
dtype0b
IdentityIdentityRead/ReadVariableOp:value:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: e
Read_1/DisableCopyOnReadDisableCopyOnRead read_1_disablecopyonread_conv2_1*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp read_1_disablecopyonread_conv2_1^Read_1/DisableCopyOnRead*&
_output_shapes
:  *
dtype0f

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*&
_output_shapes
:  k

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*&
_output_shapes
:  e
Read_2/DisableCopyOnReadDisableCopyOnRead read_2_disablecopyonread_conv2_2*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp read_2_disablecopyonread_conv2_2^Read_2/DisableCopyOnRead*&
_output_shapes
:  *
dtype0f

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:  k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:  e
Read_3/DisableCopyOnReadDisableCopyOnRead read_3_disablecopyonread_conv3_1*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp read_3_disablecopyonread_conv3_1^Read_3/DisableCopyOnRead*'
_output_shapes
: �*
dtype0g

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*'
_output_shapes
: �l

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*'
_output_shapes
: �e
Read_4/DisableCopyOnReadDisableCopyOnRead read_4_disablecopyonread_conv3_2*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp read_4_disablecopyonread_conv3_2^Read_4/DisableCopyOnRead*'
_output_shapes
: �*
dtype0g

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*'
_output_shapes
: �l

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*'
_output_shapes
: �a
Read_5/DisableCopyOnReadDisableCopyOnReadread_5_disablecopyonread_fc1*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOpread_5_disablecopyonread_fc1^Read_5/DisableCopyOnRead*!
_output_shapes
:���*
dtype0b
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*!
_output_shapes
:���h
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*!
_output_shapes
:���a
Read_6/DisableCopyOnReadDisableCopyOnReadread_6_disablecopyonread_fc2*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpread_6_disablecopyonread_fc2^Read_6/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0a
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��a
Read_7/DisableCopyOnReadDisableCopyOnReadread_7_disablecopyonread_fc3*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpread_7_disablecopyonread_fc3^Read_7/DisableCopyOnRead*
_output_shapes
:	�
*
dtype0`
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�
f
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
c
Read_8/DisableCopyOnReadDisableCopyOnReadread_8_disablecopyonread_bias1*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpread_8_disablecopyonread_bias1^Read_8/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:�c
Read_9/DisableCopyOnReadDisableCopyOnReadread_9_disablecopyonread_bias2*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpread_9_disablecopyonread_bias2^Read_9/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:�e
Read_10/DisableCopyOnReadDisableCopyOnReadread_10_disablecopyonread_bias3*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpread_10_disablecopyonread_bias3^Read_10/DisableCopyOnRead*
_output_shapes
:
*
dtype0\
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes
:
a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B conv1/.ATTRIBUTES/VARIABLE_VALUEB"conv2_1/.ATTRIBUTES/VARIABLE_VALUEB"conv2_2/.ATTRIBUTES/VARIABLE_VALUEB"conv3_1/.ATTRIBUTES/VARIABLE_VALUEB"conv3_2/.ATTRIBUTES/VARIABLE_VALUEBfc1/.ATTRIBUTES/VARIABLE_VALUEBfc2/.ATTRIBUTES/VARIABLE_VALUEBfc3/.ATTRIBUTES/VARIABLE_VALUEB bias1/.ATTRIBUTES/VARIABLE_VALUEB bias2/.ATTRIBUTES/VARIABLE_VALUEB bias3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_22Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_23IdentityIdentity_22:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_23Identity_23:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_user_specified_nameconv1:'#
!
_user_specified_name	conv2_1:'#
!
_user_specified_name	conv2_2:'#
!
_user_specified_name	conv3_1:'#
!
_user_specified_name	conv3_2:#

_user_specified_namefc1:#

_user_specified_namefc2:#

_user_specified_namefc3:%	!

_user_specified_namebias1:%
!

_user_specified_namebias2:%!

_user_specified_namebias3:=9

_output_shapes
: 

_user_specified_nameConst
�8
�
__inference___call___2225096
x8
conv2d_readvariableop_resource: :
 conv2d_1_readvariableop_resource:  :
 conv2d_2_readvariableop_resource:  ;
 conv2d_3_readvariableop_resource: �;
 conv2d_4_readvariableop_resource: �3
matmul_readvariableop_resource:���*
add_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��,
add_1_readvariableop_resource:	�3
 matmul_2_readvariableop_resource:	�
+
add_2_readvariableop_resource:

identity��Conv2D/ReadVariableOp�Conv2D_1/ReadVariableOp�Conv2D_2/ReadVariableOp�Conv2D_3/ReadVariableOp�Conv2D_4/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�MatMul_2/ReadVariableOp�add/ReadVariableOp�add_1/ReadVariableOp�add_2/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DxConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
i
ReluReluConv2D:output:0*
T0*A
_output_shapes/
-:+��������������������������� �
	MaxPool2dMaxPoolRelu:activations:0*A
_output_shapes/
-:+��������������������������� *
ksize
*
paddingSAME*
strides
�
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2D_1Conv2DMaxPool2d:output:0Conv2D_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
m
Relu_1ReluConv2D_1:output:0*
T0*A
_output_shapes/
-:+��������������������������� �
MaxPool2d_1MaxPoolRelu_1:activations:0*A
_output_shapes/
-:+��������������������������� *
ksize
*
paddingSAME*
strides
�
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2D_2Conv2DMaxPool2d:output:0Conv2D_2/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
m
Relu_2ReluConv2D_2:output:0*
T0*A
_output_shapes/
-:+��������������������������� �
MaxPool2d_2MaxPoolRelu_2:activations:0*A
_output_shapes/
-:+��������������������������� *
ksize
*
paddingSAME*
strides
�
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*'
_output_shapes
: �*
dtype0�
Conv2D_3Conv2DMaxPool2d_1:output:0Conv2D_3/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
n
Relu_3ReluConv2D_3:output:0*
T0*B
_output_shapes0
.:,�����������������������������
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*'
_output_shapes
: �*
dtype0�
Conv2D_4Conv2DMaxPool2d_2:output:0Conv2D_4/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
n
Relu_4ReluConv2D_4:output:0*
T0*B
_output_shapes0
.:,����������������������������M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2Relu_3:activations:0Relu_4:activations:0concat/axis:output:0*
N*
T0*B
_output_shapes0
.:,����������������������������^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� b  o
ReshapeReshapeconcat:output:0Reshape/shape:output:0*
T0*)
_output_shapes
:�����������w
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0t
MatMulMatMulReshape:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������J
Relu_5Reluadd:z:0*
T0*(
_output_shapes
:����������z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0|
MatMul_1MatMulRelu_5:activations:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0s
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������L
Relu_6Relu	add_1:z:0*
T0*(
_output_shapes
:����������y
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	�
*
dtype0{
MatMul_2MatMulRelu_6:activations:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
n
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:
*
dtype0r
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
X
IdentityIdentity	add_2:z:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^Conv2D_3/ReadVariableOp^Conv2D_4/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : : : : : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp22
Conv2D_3/ReadVariableOpConv2D_3/ReadVariableOp22
Conv2D_4/ReadVariableOpConv2D_4/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp:; 7

_output_shapes
:

_user_specified_namex:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�4
�
__inference___call___2224333
x8
conv2d_readvariableop_resource: :
 conv2d_1_readvariableop_resource:  :
 conv2d_2_readvariableop_resource:  ;
 conv2d_3_readvariableop_resource: �;
 conv2d_4_readvariableop_resource: �3
matmul_readvariableop_resource:���*
add_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��,
add_1_readvariableop_resource:	�3
 matmul_2_readvariableop_resource:	�
+
add_2_readvariableop_resource:

identity��Conv2D/ReadVariableOp�Conv2D_1/ReadVariableOp�Conv2D_2/ReadVariableOp�Conv2D_3/ReadVariableOp�Conv2D_4/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�MatMul_2/ReadVariableOp�add/ReadVariableOp�add_1/ReadVariableOp�add_2/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DxConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:� *
paddingSAME*
strides
O
ReluReluConv2D:output:0*
T0*'
_output_shapes
:� �
	MaxPool2dMaxPoolRelu:activations:0*'
_output_shapes
:� *
ksize
*
paddingSAME*
strides
�
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2D_1Conv2DMaxPool2d:output:0Conv2D_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:� *
paddingSAME*
strides
S
Relu_1ReluConv2D_1:output:0*
T0*'
_output_shapes
:� �
MaxPool2d_1MaxPoolRelu_1:activations:0*'
_output_shapes
:� *
ksize
*
paddingSAME*
strides
�
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2D_2Conv2DMaxPool2d:output:0Conv2D_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:� *
paddingSAME*
strides
S
Relu_2ReluConv2D_2:output:0*
T0*'
_output_shapes
:� �
MaxPool2d_2MaxPoolRelu_2:activations:0*'
_output_shapes
:� *
ksize
*
paddingSAME*
strides
�
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*'
_output_shapes
: �*
dtype0�
Conv2D_3Conv2DMaxPool2d_1:output:0Conv2D_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
T
Relu_3ReluConv2D_3:output:0*
T0*(
_output_shapes
:���
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*'
_output_shapes
: �*
dtype0�
Conv2D_4Conv2DMaxPool2d_2:output:0Conv2D_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:��*
paddingSAME*
strides
T
Relu_4ReluConv2D_4:output:0*
T0*(
_output_shapes
:��M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2Relu_3:activations:0Relu_4:activations:0concat/axis:output:0*
N*
T0*(
_output_shapes
:��^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� b  g
ReshapeReshapeconcat:output:0Reshape/shape:output:0*
T0*!
_output_shapes
:���w
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0l
MatMulMatMulReshape:output:0MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0e
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��B
Relu_5Reluadd:z:0*
T0* 
_output_shapes
:
��z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0t
MatMul_1MatMulRelu_5:activations:0MatMul_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0k
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��D
Relu_6Relu	add_1:z:0*
T0* 
_output_shapes
:
��y
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	�
*
dtype0s
MatMul_2MatMulRelu_6:activations:0MatMul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�
n
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:
*
dtype0j
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�
P
IdentityIdentity	add_2:z:0^NoOp*
T0*
_output_shapes
:	�
�
NoOpNoOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^Conv2D_3/ReadVariableOp^Conv2D_4/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�: : : : : : : : : : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp22
Conv2D_3/ReadVariableOpConv2D_3/ReadVariableOp22
Conv2D_4/ReadVariableOpConv2D_4/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp:J F
'
_output_shapes
:�

_user_specified_namex:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�8
�
__inference___call___2225017
x8
conv2d_readvariableop_resource: :
 conv2d_1_readvariableop_resource:  :
 conv2d_2_readvariableop_resource:  ;
 conv2d_3_readvariableop_resource: �;
 conv2d_4_readvariableop_resource: �3
matmul_readvariableop_resource:���*
add_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��,
add_1_readvariableop_resource:	�3
 matmul_2_readvariableop_resource:	�
+
add_2_readvariableop_resource:

identity��Conv2D/ReadVariableOp�Conv2D_1/ReadVariableOp�Conv2D_2/ReadVariableOp�Conv2D_3/ReadVariableOp�Conv2D_4/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�MatMul_2/ReadVariableOp�add/ReadVariableOp�add_1/ReadVariableOp�add_2/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DxConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
i
ReluReluConv2D:output:0*
T0*A
_output_shapes/
-:+��������������������������� �
	MaxPool2dMaxPoolRelu:activations:0*A
_output_shapes/
-:+��������������������������� *
ksize
*
paddingSAME*
strides
�
Conv2D_1/ReadVariableOpReadVariableOp conv2d_1_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2D_1Conv2DMaxPool2d:output:0Conv2D_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
m
Relu_1ReluConv2D_1:output:0*
T0*A
_output_shapes/
-:+��������������������������� �
MaxPool2d_1MaxPoolRelu_1:activations:0*A
_output_shapes/
-:+��������������������������� *
ksize
*
paddingSAME*
strides
�
Conv2D_2/ReadVariableOpReadVariableOp conv2d_2_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2D_2Conv2DMaxPool2d:output:0Conv2D_2/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
m
Relu_2ReluConv2D_2:output:0*
T0*A
_output_shapes/
-:+��������������������������� �
MaxPool2d_2MaxPoolRelu_2:activations:0*A
_output_shapes/
-:+��������������������������� *
ksize
*
paddingSAME*
strides
�
Conv2D_3/ReadVariableOpReadVariableOp conv2d_3_readvariableop_resource*'
_output_shapes
: �*
dtype0�
Conv2D_3Conv2DMaxPool2d_1:output:0Conv2D_3/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
n
Relu_3ReluConv2D_3:output:0*
T0*B
_output_shapes0
.:,�����������������������������
Conv2D_4/ReadVariableOpReadVariableOp conv2d_4_readvariableop_resource*'
_output_shapes
: �*
dtype0�
Conv2D_4Conv2DMaxPool2d_2:output:0Conv2D_4/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
n
Relu_4ReluConv2D_4:output:0*
T0*B
_output_shapes0
.:,����������������������������M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2Relu_3:activations:0Relu_4:activations:0concat/axis:output:0*
N*
T0*B
_output_shapes0
.:,����������������������������^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� b  o
ReshapeReshapeconcat:output:0Reshape/shape:output:0*
T0*)
_output_shapes
:�����������w
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0t
MatMulMatMulReshape:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������J
Relu_5Reluadd:z:0*
T0*(
_output_shapes
:����������z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0|
MatMul_1MatMulRelu_5:activations:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0s
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������L
Relu_6Relu	add_1:z:0*
T0*(
_output_shapes
:����������y
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	�
*
dtype0{
MatMul_2MatMulRelu_6:activations:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
n
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:
*
dtype0r
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
X
IdentityIdentity	add_2:z:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Conv2D/ReadVariableOp^Conv2D_1/ReadVariableOp^Conv2D_2/ReadVariableOp^Conv2D_3/ReadVariableOp^Conv2D_4/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : : : : : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp22
Conv2D_1/ReadVariableOpConv2D_1/ReadVariableOp22
Conv2D_2/ReadVariableOpConv2D_2/ReadVariableOp22
Conv2D_3/ReadVariableOpConv2D_3/ReadVariableOp22
Conv2D_4/ReadVariableOpConv2D_4/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp:; 7

_output_shapes
:

_user_specified_namex:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�3
�
#__inference__traced_restore_2225226
file_prefix0
assignvariableop_conv1: 4
assignvariableop_1_conv2_1:  4
assignvariableop_2_conv2_2:  5
assignvariableop_3_conv3_1: �5
assignvariableop_4_conv3_2: �+
assignvariableop_5_fc1:���*
assignvariableop_6_fc2:
��)
assignvariableop_7_fc3:	�
'
assignvariableop_8_bias1:	�'
assignvariableop_9_bias2:	�'
assignvariableop_10_bias3:

identity_12��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B conv1/.ATTRIBUTES/VARIABLE_VALUEB"conv2_1/.ATTRIBUTES/VARIABLE_VALUEB"conv2_2/.ATTRIBUTES/VARIABLE_VALUEB"conv3_1/.ATTRIBUTES/VARIABLE_VALUEB"conv3_2/.ATTRIBUTES/VARIABLE_VALUEBfc1/.ATTRIBUTES/VARIABLE_VALUEBfc2/.ATTRIBUTES/VARIABLE_VALUEBfc3/.ATTRIBUTES/VARIABLE_VALUEB bias1/.ATTRIBUTES/VARIABLE_VALUEB bias2/.ATTRIBUTES/VARIABLE_VALUEB bias3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv1Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2_1Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_2Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv3_1Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_conv3_2Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_fc1Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_fc2Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_fc3Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_bias1Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_bias2Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_bias3Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_12IdentityIdentity_11:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_12Identity_12:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
: : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_user_specified_nameconv1:'#
!
_user_specified_name	conv2_1:'#
!
_user_specified_name	conv2_2:'#
!
_user_specified_name	conv3_1:'#
!
_user_specified_name	conv3_2:#

_user_specified_namefc1:#

_user_specified_namefc2:#

_user_specified_namefc3:%	!

_user_specified_namebias1:%
!

_user_specified_namebias2:%!

_user_specified_namebias3"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default|
 
x
serving_default_x:0<
output_00
StatefulPartitionedCall:0���������
tensorflow/serving/predict:�
�
	conv1
conv2_1
conv2_2
conv3_1
conv3_2
fc1
fc2
fc3
		bias1
	
bias2
	bias3
__call__

signatures"
_generic_user_object
: 2conv1
!:  2conv2_1
!:  2conv2_2
":  �2conv3_1
":  �2conv3_2
:���2fc1
:
��2fc2
:	�
2fc3
:�2bias1
:�2bias2
:
2bias3
�
trace_0
trace_1
trace_2
trace_32�
__inference___call___2224333
__inference___call___2224384
__inference___call___2224435
__inference___call___2225096�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
,
serving_default"
signature_map
�B�
__inference___call___2224333x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference___call___2224384x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference___call___2224435x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference___call___2225096x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_2225045x"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
jx
kwonlydefaults
 
annotations� *
 t
__inference___call___2224333T	
*�'
 �
�
x�
� "�
unknown	�
r
__inference___call___2224384R	
)�&
�
�
x`
� "�
unknown`
t
__inference___call___2224435T	
*�'
 �
�
x�N
� "�
unknown	�N
m
__inference___call___2225096M	
�
�
�	
x
� "!�
unknown���������
�
%__inference_signature_wrapper_2225045d	
 �
� 
�

x�	
x"3�0
.
output_0"�
output_0���������
