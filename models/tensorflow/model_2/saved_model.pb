??

??
.
Abs
x"T
y"T"
Ttype:

2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
?
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
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
?
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ??
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-0-g3f878cff5b68??
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
p
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametrue_positives
i
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
: *
dtype0
r
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namefalse_positives
k
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
: *
dtype0
r
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namefalse_negatives
k
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
: *
dtype0
|
weights_intermediateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameweights_intermediate
u
(weights_intermediate/Read/ReadVariableOpReadVariableOpweights_intermediate*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
??*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:@*
dtype0
?
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/m
?
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/m
?
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:@*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
??*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:@*
dtype0
?
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/v
?
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/v
?
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:@*
dtype0

NoOpNoOp
?@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?@
value?@B?@ B?@
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	keras_api* 

	keras_api* 
?

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*
?
%iter

&beta_1

'beta_2
	(decay
)learning_ratem?m?*m?+m?,m?-m?v?v?*v?+v?,v?-v?*
<
*0
+1
,2
-3
.4
/5
6
7*
.
*0
+1
,2
-3
4
5*
* 
?
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

5serving_default* 
* 
?

*kernel
+bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*
?
<axis
	,gamma
-beta
.moving_mean
/moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
.
*0
+1
,2
-3
.4
/5*
 
*0
+1
,2
-3*
* 
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbatch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEbatch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*
.
0
1
2
3
4
5*

Y0
Z1
[2*
* 
* 
* 

*0
+1*

*0
+1*
* 
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
* 
 
,0
-1
.2
/3*

,0
-1*
* 
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 
* 
* 

.0
/1*
'
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
* 
8
	ptotal
	qcount
r	variables
s	keras_api*
H
	ttotal
	ucount
v
_fn_kwargs
w	variables
x	keras_api*
?
y
init_shape
ztrue_positives
{false_positives
|false_negatives
}weights_intermediate
~	variables
	keras_api*
* 
* 
* 
* 
* 

.0
/1*
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
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

p0
q1*

r	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

t0
u1*

w	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEweights_intermediateCkeras_api/metrics/2/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUE*
 
z0
{1
|2
}3*

~	variables*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/batch_normalization/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/batch_normalization/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/batch_normalization/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/batch_normalization/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
serving_default_input_2Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_85515
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp(weights_intermediate/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
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
GPU2*0J 8? *'
f"R 
__inference__traced_save_85851
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancetotalcounttotal_1count_1true_positivesfalse_positivesfalse_negativesweights_intermediateAdam/dense/kernel/mAdam/dense/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/dense/kernel/vAdam/dense/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/v*-
Tin&
$2"*
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
GPU2*0J 8? **
f%R#
!__inference__traced_restore_85960??
?
?
@__inference_model_layer_call_and_return_conditional_losses_85577

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????mm@?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????mm@:@:@:@:@:*
epsilon%o?:*
is_training( ?
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? D ?
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????i
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*)
_output_shapes
:????????????
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????: : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
'__inference_model_2_layer_call_fn_85357
inputs_0
inputs_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:
??
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_85138o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
C
'__inference_flatten_layer_call_fn_85722

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_84944b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????$$@:W S
/
_output_shapes
:?????????$$@
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84850

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
I
-__inference_max_pooling2d_layer_call_fn_85712

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_84901?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_85707

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
@__inference_model_layer_call_and_return_conditional_losses_85089
input_3&
conv2d_85072:@
conv2d_85074:@'
batch_normalization_85077:@'
batch_normalization_85079:@'
batch_normalization_85081:@'
batch_normalization_85083:@
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_85072conv2d_85074*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_84922?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_85077batch_normalization_85079batch_normalization_85081batch_normalization_85083*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84881?
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_84901?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_84944q
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:????????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
?
@__inference_model_layer_call_and_return_conditional_losses_84947

inputs&
conv2d_84923:@
conv2d_84925:@'
batch_normalization_84928:@'
batch_normalization_84930:@'
batch_normalization_84932:@'
batch_normalization_84934:@
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_84923conv2d_84925*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_84922?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_84928batch_normalization_84930batch_normalization_84932batch_normalization_84934*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84850?
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_84901?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_84944q
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:????????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?I
?

B__inference_model_2_layer_call_and_return_conditional_losses_85435
inputs_0
inputs_1E
+model_conv2d_conv2d_readvariableop_resource:@:
,model_conv2d_biasadd_readvariableop_resource:@?
1model_batch_normalization_readvariableop_resource:@A
3model_batch_normalization_readvariableop_1_resource:@P
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource:@R
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@8
$dense_matmul_readvariableop_resource:
??3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?(model/batch_normalization/ReadVariableOp?*model/batch_normalization/ReadVariableOp_1?*model/batch_normalization/ReadVariableOp_2?*model/batch_normalization/ReadVariableOp_3?#model/conv2d/BiasAdd/ReadVariableOp?%model/conv2d/BiasAdd_1/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?$model/conv2d/Conv2D_1/ReadVariableOp?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
model/conv2d/Conv2DConv2Dinputs_0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@r
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????mm@?
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0?
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3model/conv2d/Relu:activations:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????mm@:@:@:@:@:*
epsilon%o?:*
is_training( ?
model/max_pooling2d/MaxPoolMaxPool.model/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? D ?
model/flatten/ReshapeReshape$model/max_pooling2d/MaxPool:output:0model/flatten/Const:output:0*
T0*)
_output_shapes
:????????????
$model/conv2d/Conv2D_1/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
model/conv2d/Conv2D_1Conv2Dinputs_1,model/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
?
%model/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/conv2d/BiasAdd_1BiasAddmodel/conv2d/Conv2D_1:output:0-model/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@v
model/conv2d/Relu_1Relumodel/conv2d/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????mm@?
*model/batch_normalization/ReadVariableOp_2ReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0?
*model/batch_normalization/ReadVariableOp_3ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
,model/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3!model/conv2d/Relu_1:activations:02model/batch_normalization/ReadVariableOp_2:value:02model/batch_normalization/ReadVariableOp_3:value:0Cmodel/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Emodel/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????mm@:@:@:@:@:*
epsilon%o?:*
is_training( ?
model/max_pooling2d/MaxPool_1MaxPool0model/batch_normalization/FusedBatchNormV3_1:y:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
f
model/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"???? D ?
model/flatten/Reshape_1Reshape&model/max_pooling2d/MaxPool_1:output:0model/flatten/Const_1:output:0*
T0*)
_output_shapes
:????????????
tf.math.subtract/SubSubmodel/flatten/Reshape:output:0 model/flatten/Reshape_1:output:0*
T0*)
_output_shapes
:???????????d
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*)
_output_shapes
:????????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMultf.math.abs/Abs:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1<^model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp>^model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1+^model/batch_normalization/ReadVariableOp_2+^model/batch_normalization/ReadVariableOp_3$^model/conv2d/BiasAdd/ReadVariableOp&^model/conv2d/BiasAdd_1/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp%^model/conv2d/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12z
;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2~
=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12X
*model/batch_normalization/ReadVariableOp_2*model/batch_normalization/ReadVariableOp_22X
*model/batch_normalization/ReadVariableOp_3*model/batch_normalization/ReadVariableOp_32J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2N
%model/conv2d/BiasAdd_1/ReadVariableOp%model/conv2d/BiasAdd_1/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2L
$model/conv2d/Conv2D_1/ReadVariableOp$model/conv2d/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?

?
@__inference_dense_layer_call_and_return_conditional_losses_85131

inputs2
matmul_readvariableop_resource:
??-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_85532

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_84947q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?Y
?
B__inference_model_2_layer_call_and_return_conditional_losses_85491
inputs_0
inputs_1E
+model_conv2d_conv2d_readvariableop_resource:@:
,model_conv2d_biasadd_readvariableop_resource:@?
1model_batch_normalization_readvariableop_resource:@A
3model_batch_normalization_readvariableop_1_resource:@P
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource:@R
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@8
$dense_matmul_readvariableop_resource:
??3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?(model/batch_normalization/AssignNewValue?*model/batch_normalization/AssignNewValue_1?*model/batch_normalization/AssignNewValue_2?*model/batch_normalization/AssignNewValue_3?9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?(model/batch_normalization/ReadVariableOp?*model/batch_normalization/ReadVariableOp_1?*model/batch_normalization/ReadVariableOp_2?*model/batch_normalization/ReadVariableOp_3?#model/conv2d/BiasAdd/ReadVariableOp?%model/conv2d/BiasAdd_1/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?$model/conv2d/Conv2D_1/ReadVariableOp?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
model/conv2d/Conv2DConv2Dinputs_0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@r
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????mm@?
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0?
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3model/conv2d/Relu:activations:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????mm@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
(model/batch_normalization/AssignNewValueAssignVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource7model/batch_normalization/FusedBatchNormV3:batch_mean:0:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
*model/batch_normalization/AssignNewValue_1AssignVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource;model/batch_normalization/FusedBatchNormV3:batch_variance:0<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
model/max_pooling2d/MaxPoolMaxPool.model/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? D ?
model/flatten/ReshapeReshape$model/max_pooling2d/MaxPool:output:0model/flatten/Const:output:0*
T0*)
_output_shapes
:????????????
$model/conv2d/Conv2D_1/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
model/conv2d/Conv2D_1Conv2Dinputs_1,model/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
?
%model/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/conv2d/BiasAdd_1BiasAddmodel/conv2d/Conv2D_1:output:0-model/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@v
model/conv2d/Relu_1Relumodel/conv2d/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????mm@?
*model/batch_normalization/ReadVariableOp_2ReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0?
*model/batch_normalization/ReadVariableOp_3ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource)^model/batch_normalization/AssignNewValue*
_output_shapes
:@*
dtype0?
=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource+^model/batch_normalization/AssignNewValue_1*
_output_shapes
:@*
dtype0?
,model/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3!model/conv2d/Relu_1:activations:02model/batch_normalization/ReadVariableOp_2:value:02model/batch_normalization/ReadVariableOp_3:value:0Cmodel/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Emodel/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????mm@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
*model/batch_normalization/AssignNewValue_2AssignVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource9model/batch_normalization/FusedBatchNormV3_1:batch_mean:0)^model/batch_normalization/AssignNewValue<^model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0?
*model/batch_normalization/AssignNewValue_3AssignVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource=model/batch_normalization/FusedBatchNormV3_1:batch_variance:0+^model/batch_normalization/AssignNewValue_1>^model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0?
model/max_pooling2d/MaxPool_1MaxPool0model/batch_normalization/FusedBatchNormV3_1:y:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
f
model/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"???? D ?
model/flatten/Reshape_1Reshape&model/max_pooling2d/MaxPool_1:output:0model/flatten/Const_1:output:0*
T0*)
_output_shapes
:????????????
tf.math.subtract/SubSubmodel/flatten/Reshape:output:0 model/flatten/Reshape_1:output:0*
T0*)
_output_shapes
:???????????d
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*)
_output_shapes
:????????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMultf.math.abs/Abs:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp)^model/batch_normalization/AssignNewValue+^model/batch_normalization/AssignNewValue_1+^model/batch_normalization/AssignNewValue_2+^model/batch_normalization/AssignNewValue_3:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1<^model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp>^model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1+^model/batch_normalization/ReadVariableOp_2+^model/batch_normalization/ReadVariableOp_3$^model/conv2d/BiasAdd/ReadVariableOp&^model/conv2d/BiasAdd_1/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp%^model/conv2d/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2T
(model/batch_normalization/AssignNewValue(model/batch_normalization/AssignNewValue2X
*model/batch_normalization/AssignNewValue_1*model/batch_normalization/AssignNewValue_12X
*model/batch_normalization/AssignNewValue_2*model/batch_normalization/AssignNewValue_22X
*model/batch_normalization/AssignNewValue_3*model/batch_normalization/AssignNewValue_32v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12z
;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2~
=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12X
*model/batch_normalization/ReadVariableOp_2*model/batch_normalization/ReadVariableOp_22X
*model/batch_normalization/ReadVariableOp_3*model/batch_normalization/ReadVariableOp_32J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2N
%model/conv2d/BiasAdd_1/ReadVariableOp%model/conv2d/BiasAdd_1/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2L
$model/conv2d/Conv2D_1/ReadVariableOp$model/conv2d/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
B__inference_model_2_layer_call_and_return_conditional_losses_85138

inputs
inputs_1%
model_85098:@
model_85100:@
model_85102:@
model_85104:@
model_85106:@
model_85108:@
dense_85132:
??
dense_85134:
identity??dense/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_85098model_85100model_85102model_85104model_85106model_85108*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_84947?
model/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_85098model_85100model_85102model_85104model_85106model_85108*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_84947?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*)
_output_shapes
:???????????d
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*)
_output_shapes
:????????????
dense/StatefulPartitionedCallStatefulPartitionedCalltf.math.abs/Abs:y:0dense_85132dense_85134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_85131u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_85515
input_1
input_2!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:
??
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_84828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?

?
'__inference_model_2_layer_call_fn_85265
input_1
input_2!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:
??
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_85224o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
B__inference_model_2_layer_call_and_return_conditional_losses_85329
input_1
input_2%
model_85301:@
model_85303:@
model_85305:@
model_85307:@
model_85309:@
model_85311:@
dense_85323:
??
dense_85325:
identity??dense/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_85301model_85303model_85305model_85307model_85309model_85311*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_85017?
model/StatefulPartitionedCall_1StatefulPartitionedCallinput_2model_85301model_85303model_85305model_85307model_85309model_85311^model/StatefulPartitionedCall*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_85017?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*)
_output_shapes
:???????????d
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*)
_output_shapes
:????????????
dense/StatefulPartitionedCallStatefulPartitionedCalltf.math.abs/Abs:y:0dense_85323dense_85325*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_85131u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?

?
'__inference_model_2_layer_call_fn_85379
inputs_0
inputs_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:
??
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_85224o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?D
?
__inference__traced_save_85851
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop3
/savev2_weights_intermediate_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/2/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop/savev2_weights_intermediate_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:: : : : : :@:@:@:@:@:@: : : : : : : : :
??::@:@:@:@:
??::@:@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
??: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:  

_output_shapes
:@: !

_output_shapes
:@:"

_output_shapes
: 
?	
?
3__inference_batch_normalization_layer_call_fn_85658

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84850?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?&
?
@__inference_model_layer_call_and_return_conditional_losses_85605

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????mm@?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????mm@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? D ?
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????i
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*)
_output_shapes
:????????????
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????: : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?R
?
 __inference__wrapped_model_84828
input_1
input_2M
3model_2_model_conv2d_conv2d_readvariableop_resource:@B
4model_2_model_conv2d_biasadd_readvariableop_resource:@G
9model_2_model_batch_normalization_readvariableop_resource:@I
;model_2_model_batch_normalization_readvariableop_1_resource:@X
Jmodel_2_model_batch_normalization_fusedbatchnormv3_readvariableop_resource:@Z
Lmodel_2_model_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@@
,model_2_dense_matmul_readvariableop_resource:
??;
-model_2_dense_biasadd_readvariableop_resource:
identity??$model_2/dense/BiasAdd/ReadVariableOp?#model_2/dense/MatMul/ReadVariableOp?Amodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp?Cmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?Cmodel_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?Emodel_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?0model_2/model/batch_normalization/ReadVariableOp?2model_2/model/batch_normalization/ReadVariableOp_1?2model_2/model/batch_normalization/ReadVariableOp_2?2model_2/model/batch_normalization/ReadVariableOp_3?+model_2/model/conv2d/BiasAdd/ReadVariableOp?-model_2/model/conv2d/BiasAdd_1/ReadVariableOp?*model_2/model/conv2d/Conv2D/ReadVariableOp?,model_2/model/conv2d/Conv2D_1/ReadVariableOp?
*model_2/model/conv2d/Conv2D/ReadVariableOpReadVariableOp3model_2_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
model_2/model/conv2d/Conv2DConv2Dinput_12model_2/model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
?
+model_2/model/conv2d/BiasAdd/ReadVariableOpReadVariableOp4model_2_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_2/model/conv2d/BiasAddBiasAdd$model_2/model/conv2d/Conv2D:output:03model_2/model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@?
model_2/model/conv2d/ReluRelu%model_2/model/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????mm@?
0model_2/model/batch_normalization/ReadVariableOpReadVariableOp9model_2_model_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0?
2model_2/model/batch_normalization/ReadVariableOp_1ReadVariableOp;model_2_model_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Amodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpJmodel_2_model_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Cmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLmodel_2_model_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
2model_2/model/batch_normalization/FusedBatchNormV3FusedBatchNormV3'model_2/model/conv2d/Relu:activations:08model_2/model/batch_normalization/ReadVariableOp:value:0:model_2/model/batch_normalization/ReadVariableOp_1:value:0Imodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Kmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????mm@:@:@:@:@:*
epsilon%o?:*
is_training( ?
#model_2/model/max_pooling2d/MaxPoolMaxPool6model_2/model/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
l
model_2/model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? D ?
model_2/model/flatten/ReshapeReshape,model_2/model/max_pooling2d/MaxPool:output:0$model_2/model/flatten/Const:output:0*
T0*)
_output_shapes
:????????????
,model_2/model/conv2d/Conv2D_1/ReadVariableOpReadVariableOp3model_2_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
model_2/model/conv2d/Conv2D_1Conv2Dinput_24model_2/model/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
?
-model_2/model/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp4model_2_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_2/model/conv2d/BiasAdd_1BiasAdd&model_2/model/conv2d/Conv2D_1:output:05model_2/model/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@?
model_2/model/conv2d/Relu_1Relu'model_2/model/conv2d/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????mm@?
2model_2/model/batch_normalization/ReadVariableOp_2ReadVariableOp9model_2_model_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0?
2model_2/model/batch_normalization/ReadVariableOp_3ReadVariableOp;model_2_model_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Cmodel_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpJmodel_2_model_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Emodel_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpLmodel_2_model_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
4model_2/model/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3)model_2/model/conv2d/Relu_1:activations:0:model_2/model/batch_normalization/ReadVariableOp_2:value:0:model_2/model/batch_normalization/ReadVariableOp_3:value:0Kmodel_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Mmodel_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????mm@:@:@:@:@:*
epsilon%o?:*
is_training( ?
%model_2/model/max_pooling2d/MaxPool_1MaxPool8model_2/model/batch_normalization/FusedBatchNormV3_1:y:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
n
model_2/model/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"???? D ?
model_2/model/flatten/Reshape_1Reshape.model_2/model/max_pooling2d/MaxPool_1:output:0&model_2/model/flatten/Const_1:output:0*
T0*)
_output_shapes
:????????????
model_2/tf.math.subtract/SubSub&model_2/model/flatten/Reshape:output:0(model_2/model/flatten/Reshape_1:output:0*
T0*)
_output_shapes
:???????????t
model_2/tf.math.abs/AbsAbs model_2/tf.math.subtract/Sub:z:0*
T0*)
_output_shapes
:????????????
#model_2/dense/MatMul/ReadVariableOpReadVariableOp,model_2_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model_2/dense/MatMulMatMulmodel_2/tf.math.abs/Abs:y:0+model_2/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model_2/dense/BiasAdd/ReadVariableOpReadVariableOp-model_2_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_2/dense/BiasAddBiasAddmodel_2/dense/MatMul:product:0,model_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
model_2/dense/SigmoidSigmoidmodel_2/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitymodel_2/dense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp%^model_2/dense/BiasAdd/ReadVariableOp$^model_2/dense/MatMul/ReadVariableOpB^model_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOpD^model_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1D^model_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOpF^model_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_11^model_2/model/batch_normalization/ReadVariableOp3^model_2/model/batch_normalization/ReadVariableOp_13^model_2/model/batch_normalization/ReadVariableOp_23^model_2/model/batch_normalization/ReadVariableOp_3,^model_2/model/conv2d/BiasAdd/ReadVariableOp.^model_2/model/conv2d/BiasAdd_1/ReadVariableOp+^model_2/model/conv2d/Conv2D/ReadVariableOp-^model_2/model/conv2d/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 2L
$model_2/dense/BiasAdd/ReadVariableOp$model_2/dense/BiasAdd/ReadVariableOp2J
#model_2/dense/MatMul/ReadVariableOp#model_2/dense/MatMul/ReadVariableOp2?
Amodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOpAmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Cmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Cmodel_2/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
Cmodel_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOpCmodel_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2?
Emodel_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1Emodel_2/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12d
0model_2/model/batch_normalization/ReadVariableOp0model_2/model/batch_normalization/ReadVariableOp2h
2model_2/model/batch_normalization/ReadVariableOp_12model_2/model/batch_normalization/ReadVariableOp_12h
2model_2/model/batch_normalization/ReadVariableOp_22model_2/model/batch_normalization/ReadVariableOp_22h
2model_2/model/batch_normalization/ReadVariableOp_32model_2/model/batch_normalization/ReadVariableOp_32Z
+model_2/model/conv2d/BiasAdd/ReadVariableOp+model_2/model/conv2d/BiasAdd/ReadVariableOp2^
-model_2/model/conv2d/BiasAdd_1/ReadVariableOp-model_2/model/conv2d/BiasAdd_1/ReadVariableOp2X
*model_2/model/conv2d/Conv2D/ReadVariableOp*model_2/model/conv2d/Conv2D/ReadVariableOp2\
,model_2/model/conv2d/Conv2D_1/ReadVariableOp,model_2/model/conv2d/Conv2D_1/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
B__inference_model_2_layer_call_and_return_conditional_losses_85297
input_1
input_2%
model_85269:@
model_85271:@
model_85273:@
model_85275:@
model_85277:@
model_85279:@
dense_85291:
??
dense_85293:
identity??dense/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_85269model_85271model_85273model_85275model_85277model_85279*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_84947?
model/StatefulPartitionedCall_1StatefulPartitionedCallinput_2model_85269model_85271model_85273model_85275model_85277model_85279*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_84947?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*)
_output_shapes
:???????????d
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*)
_output_shapes
:????????????
dense/StatefulPartitionedCallStatefulPartitionedCalltf.math.abs/Abs:y:0dense_85291dense_85293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_85131u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_85717

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_85614

inputs
unknown:
??
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_85131o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
ʂ
?
!__inference__traced_restore_85960
file_prefix1
assignvariableop_dense_kernel:
??+
assignvariableop_1_dense_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: :
 assignvariableop_7_conv2d_kernel:@,
assignvariableop_8_conv2d_bias:@:
,assignvariableop_9_batch_normalization_gamma:@:
,assignvariableop_10_batch_normalization_beta:@A
3assignvariableop_11_batch_normalization_moving_mean:@E
7assignvariableop_12_batch_normalization_moving_variance:@#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: ,
"assignvariableop_17_true_positives: -
#assignvariableop_18_false_positives: -
#assignvariableop_19_false_negatives: 2
(assignvariableop_20_weights_intermediate: ;
'assignvariableop_21_adam_dense_kernel_m:
??3
%assignvariableop_22_adam_dense_bias_m:B
(assignvariableop_23_adam_conv2d_kernel_m:@4
&assignvariableop_24_adam_conv2d_bias_m:@B
4assignvariableop_25_adam_batch_normalization_gamma_m:@A
3assignvariableop_26_adam_batch_normalization_beta_m:@;
'assignvariableop_27_adam_dense_kernel_v:
??3
%assignvariableop_28_adam_dense_bias_v:B
(assignvariableop_29_adam_conv2d_kernel_v:@4
&assignvariableop_30_adam_conv2d_bias_v:@B
4assignvariableop_31_adam_batch_normalization_gamma_v:@A
3assignvariableop_32_adam_batch_normalization_beta_v:@
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/2/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_conv2d_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_batch_normalization_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp,assignvariableop_10_batch_normalization_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp3assignvariableop_11_batch_normalization_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_batch_normalization_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_true_positivesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_false_positivesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_false_negativesIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_weights_intermediateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_conv2d_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_conv2d_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_batch_normalization_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_batch_normalization_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv2d_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv2d_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_batch_normalization_gamma_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp3assignvariableop_32_adam_batch_normalization_beta_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
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
_user_specified_namefile_prefix
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_84901

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_model_2_layer_call_and_return_conditional_losses_85224

inputs
inputs_1%
model_85196:@
model_85198:@
model_85200:@
model_85202:@
model_85204:@
model_85206:@
dense_85218:
??
dense_85220:
identity??dense/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_85196model_85198model_85200model_85202model_85204model_85206*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_85017?
model/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_85196model_85198model_85200model_85202model_85204model_85206^model/StatefulPartitionedCall*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_85017?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*)
_output_shapes
:???????????d
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*)
_output_shapes
:????????????
dense/StatefulPartitionedCallStatefulPartitionedCalltf.math.abs/Abs:y:0dense_85218dense_85220*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_85131u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
%__inference_model_layer_call_fn_84962
input_3!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_84947q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_85645

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????mm@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????mm@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_85689

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
@__inference_model_layer_call_and_return_conditional_losses_85017

inputs&
conv2d_85000:@
conv2d_85002:@'
batch_normalization_85005:@'
batch_normalization_85007:@'
batch_normalization_85009:@'
batch_normalization_85011:@
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_85000conv2d_85002*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_84922?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_85005batch_normalization_85007batch_normalization_85009batch_normalization_85011*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84881?
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_84901?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_84944q
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:????????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_85049
input_3!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_85017q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84881

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
@__inference_model_layer_call_and_return_conditional_losses_85069
input_3&
conv2d_85052:@
conv2d_85054:@'
batch_normalization_85057:@'
batch_normalization_85059:@'
batch_normalization_85061:@'
batch_normalization_85063:@
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_85052conv2d_85054*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_84922?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_85057batch_normalization_85059batch_normalization_85061batch_normalization_85063*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84850?
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_84901?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_84944q
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:????????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_84922

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????mm@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????mm@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_85549

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_85017q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_85625

inputs2
matmul_readvariableop_resource:
??-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_conv2d_layer_call_fn_85634

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_84922w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????mm@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
3__inference_batch_normalization_layer_call_fn_85671

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_84881?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
'__inference_model_2_layer_call_fn_85157
input_1
input_2!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:
??
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_85138o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_85728

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? D ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????$$@:W S
/
_output_shapes
:?????????$$@
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_84944

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? D ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????$$@:W S
/
_output_shapes
:?????????$$@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????
E
input_2:
serving_default_input_2:0???????????9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
(
	keras_api"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
?

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
?
%iter

&beta_1

'beta_2
	(decay
)learning_ratem?m?*m?+m?,m?-m?v?v?*v?+v?,v?-v?"
	optimizer
X
*0
+1
,2
-3
.4
/5
6
7"
trackable_list_wrapper
J
*0
+1
,2
-3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_model_2_layer_call_fn_85157
'__inference_model_2_layer_call_fn_85357
'__inference_model_2_layer_call_fn_85379
'__inference_model_2_layer_call_fn_85265?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_2_layer_call_and_return_conditional_losses_85435
B__inference_model_2_layer_call_and_return_conditional_losses_85491
B__inference_model_2_layer_call_and_return_conditional_losses_85297
B__inference_model_2_layer_call_and_return_conditional_losses_85329?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_84828input_1input_2"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
5serving_default"
signature_map
"
_tf_keras_input_layer
?

*kernel
+bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
?
<axis
	,gamma
-beta
.moving_mean
/moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
J
*0
+1
,2
-3
.4
/5"
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_model_layer_call_fn_84962
%__inference_model_layer_call_fn_85532
%__inference_model_layer_call_fn_85549
%__inference_model_layer_call_fn_85049?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_model_layer_call_and_return_conditional_losses_85577
@__inference_model_layer_call_and_return_conditional_losses_85605
@__inference_model_layer_call_and_return_conditional_losses_85069
@__inference_model_layer_call_and_return_conditional_losses_85089?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_generic_user_object
"
_generic_user_object
 :
??2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_dense_layer_call_fn_85614?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_85625?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%@2conv2d/kernel
:@2conv2d/bias
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
.
.0
/1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
5
Y0
Z1
[2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
#__inference_signature_wrapper_85515input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_conv2d_layer_call_fn_85634?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv2d_layer_call_and_return_conditional_losses_85645?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
,0
-1
.2
/3"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
?2?
3__inference_batch_normalization_layer_call_fn_85658
3__inference_batch_normalization_layer_call_fn_85671?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_85689
N__inference_batch_normalization_layer_call_and_return_conditional_losses_85707?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_max_pooling2d_layer_call_fn_85712?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_85717?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_flatten_layer_call_fn_85722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_85728?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
.0
/1"
trackable_list_wrapper
C
0
1
2
3
4"
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
N
	ptotal
	qcount
r	variables
s	keras_api"
_tf_keras_metric
^
	ttotal
	ucount
v
_fn_kwargs
w	variables
x	keras_api"
_tf_keras_metric
?
y
init_shape
ztrue_positives
{false_positives
|false_negatives
}weights_intermediate
~	variables
	keras_api"
_tf_keras_metric
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
.
.0
/1"
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
trackable_dict_wrapper
:  (2total
:  (2count
.
p0
q1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
t0
u1"
trackable_list_wrapper
-
w	variables"
_generic_user_object
 "
trackable_list_wrapper
:  (2true_positives
:  (2false_positives
:  (2false_negatives
 :  (2weights_intermediate
<
z0
{1
|2
}3"
trackable_list_wrapper
-
~	variables"
_generic_user_object
%:#
??2Adam/dense/kernel/m
:2Adam/dense/bias/m
,:*@2Adam/conv2d/kernel/m
:@2Adam/conv2d/bias/m
,:*@2 Adam/batch_normalization/gamma/m
+:)@2Adam/batch_normalization/beta/m
%:#
??2Adam/dense/kernel/v
:2Adam/dense/bias/v
,:*@2Adam/conv2d/kernel/v
:@2Adam/conv2d/bias/v
,:*@2 Adam/batch_normalization/gamma/v
+:)@2Adam/batch_normalization/beta/v?
 __inference__wrapped_model_84828?*+,-./l?i
b?_
]?Z
+?(
input_1???????????
+?(
input_2???????????
? "-?*
(
dense?
dense??????????
N__inference_batch_normalization_layer_call_and_return_conditional_losses_85689?,-./M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_85707?,-./M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
3__inference_batch_normalization_layer_call_fn_85658?,-./M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
3__inference_batch_normalization_layer_call_fn_85671?,-./M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
A__inference_conv2d_layer_call_and_return_conditional_losses_85645n*+9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????mm@
? ?
&__inference_conv2d_layer_call_fn_85634a*+9?6
/?,
*?'
inputs???????????
? " ??????????mm@?
@__inference_dense_layer_call_and_return_conditional_losses_85625^1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? z
%__inference_dense_layer_call_fn_85614Q1?.
'?$
"?
inputs???????????
? "???????????
B__inference_flatten_layer_call_and_return_conditional_losses_85728b7?4
-?*
(?%
inputs?????????$$@
? "'?$
?
0???????????
? ?
'__inference_flatten_layer_call_fn_85722U7?4
-?*
(?%
inputs?????????$$@
? "?????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_85717?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_85712?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_model_2_layer_call_and_return_conditional_losses_85297?*+,-./t?q
j?g
]?Z
+?(
input_1???????????
+?(
input_2???????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_85329?*+,-./t?q
j?g
]?Z
+?(
input_1???????????
+?(
input_2???????????
p

 
? "%?"
?
0?????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_85435?*+,-./v?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_85491?*+,-./v?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p

 
? "%?"
?
0?????????
? ?
'__inference_model_2_layer_call_fn_85157?*+,-./t?q
j?g
]?Z
+?(
input_1???????????
+?(
input_2???????????
p 

 
? "???????????
'__inference_model_2_layer_call_fn_85265?*+,-./t?q
j?g
]?Z
+?(
input_1???????????
+?(
input_2???????????
p

 
? "???????????
'__inference_model_2_layer_call_fn_85357?*+,-./v?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p 

 
? "???????????
'__inference_model_2_layer_call_fn_85379?*+,-./v?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p

 
? "???????????
@__inference_model_layer_call_and_return_conditional_losses_85069u*+,-./B??
8?5
+?(
input_3???????????
p 

 
? "'?$
?
0???????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_85089u*+,-./B??
8?5
+?(
input_3???????????
p

 
? "'?$
?
0???????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_85577t*+,-./A?>
7?4
*?'
inputs???????????
p 

 
? "'?$
?
0???????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_85605t*+,-./A?>
7?4
*?'
inputs???????????
p

 
? "'?$
?
0???????????
? ?
%__inference_model_layer_call_fn_84962h*+,-./B??
8?5
+?(
input_3???????????
p 

 
? "?????????????
%__inference_model_layer_call_fn_85049h*+,-./B??
8?5
+?(
input_3???????????
p

 
? "?????????????
%__inference_model_layer_call_fn_85532g*+,-./A?>
7?4
*?'
inputs???????????
p 

 
? "?????????????
%__inference_model_layer_call_fn_85549g*+,-./A?>
7?4
*?'
inputs???????????
p

 
? "?????????????
#__inference_signature_wrapper_85515?*+,-./}?z
? 
s?p
6
input_1+?(
input_1???????????
6
input_2+?(
input_2???????????"-?*
(
dense?
dense?????????