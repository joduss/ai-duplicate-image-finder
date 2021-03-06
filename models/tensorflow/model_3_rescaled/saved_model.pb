??
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
 ?"serve*2.8.02v2.8.0-0-g3f878cff5b68??
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?K*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?K*
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
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@`*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:`*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:`*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:`*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:`*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:`*
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
dtype0*
shape:	?K*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?K*
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
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:@`*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:`*
dtype0
?
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/batch_normalization_1/gamma/m
?
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:`*
dtype0
?
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!Adam/batch_normalization_1/beta/m
?
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:`*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?K*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	?K*
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
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:@`*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:`*
dtype0
?
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/batch_normalization_1/gamma/v
?
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:`*
dtype0
?
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!Adam/batch_normalization_1/beta/v
?
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:`*
dtype0

NoOpNoOp
?g
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?g
value?gB?g B?f
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
layer-8
layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*

!	keras_api* 

"	keras_api* 
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'_random_generator
(__call__
*)&call_and_return_all_conditional_losses* 
?

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*
?
2iter

3beta_1

4beta_2
	5decay*m?+m?6m?7m?8m?9m?<m?=m?>m??m?*v?+v?6v?7v?8v?9v?<v?=v?>v??v?*
j
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
*12
+13*
J
60
71
82
93
<4
=5
>6
?7
*8
+9*
* 
?
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Gserving_default* 
* 
?

6kernel
7bias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses*
?
Naxis
	8gamma
9beta
:moving_mean
;moving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses*
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y_random_generator
Z__call__
*[&call_and_return_all_conditional_losses* 
?
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
?

<kernel
=bias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses*
?
haxis
	>gamma
?beta
@moving_mean
Amoving_variance
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses*
?
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s_random_generator
t__call__
*u&call_and_return_all_conditional_losses* 
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
?
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
Z
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11*
<
60
71
82
93
<4
=5
>6
?7*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
#	variables
$trainable_variables
%regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 
* 
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
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
OI
VARIABLE_VALUEconv2d_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
 
:0
;1
@2
A3*
5
0
1
2
3
4
5
6*

?0
?1
?2*
* 
* 
* 

60
71*

60
71*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*
* 
* 
* 
 
80
91
:2
;3*

80
91*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 
* 
* 

<0
=1*

<0
=1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 
* 
 
>0
?1
@2
A3*

>0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
 
:0
;1
@2
A3*
J
0
1
2
3
4
5
6
7
8
9*
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
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
?
?
init_shape
?true_positives
?false_positives
?false_negatives
?weights_intermediate
?	variables
?	keras_api*
* 
* 
* 
* 
* 

:0
;1*
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
* 
* 

@0
A1*
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
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEweights_intermediateCkeras_api/metrics/2/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUE*
$
?0
?1
?2
?3*

?	variables*
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
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
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
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
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
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_95856
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp(weights_intermediate/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpConst*;
Tin4
220	*
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
__inference__traced_save_96490
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancetotalcounttotal_1count_1true_positivesfalse_positivesfalse_negativesweights_intermediateAdam/dense/kernel/mAdam/dense/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/dense/kernel/vAdam/dense/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/v*:
Tin3
12/*
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
!__inference__traced_restore_96638??
?
`
'__inference_dropout_layer_call_fn_96171

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_94911w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????mm@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????mm@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????mm@
 
_user_specified_nameinputs
?
?
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95196

inputs
inputs_1%
model_95131:@
model_95133:@
model_95135:@
model_95137:@
model_95139:@
model_95141:@%
model_95143:@`
model_95145:`
model_95147:`
model_95149:`
model_95151:`
model_95153:`
dense_95190:	?K
dense_95192:
identity??dense/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_95131model_95133model_95135model_95137model_95139model_95141model_95143model_95145model_95147model_95149model_95151model_95153*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94825?
model/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_95131model_95133model_95135model_95137model_95139model_95141model_95143model_95145model_95147model_95149model_95151model_95153*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94825?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:??????????Kc
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*(
_output_shapes
:??????????K?
dropout_2/PartitionedCallPartitionedCalltf.math.abs/Abs:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_95176?
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_95190dense_95192*
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
@__inference_dense_layer_call_and_return_conditional_losses_95189u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 2>
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
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_96161

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
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96280

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
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
-:+???????????????????????????`?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_96218

inputs8
conv2d_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  `i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  `w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????$$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????$$@
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_96290

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_94878w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  `22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  `
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_94813

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  `c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????  `"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  `:W S
/
_output_shapes
:?????????  `
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_96322

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_94822a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????K"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

`:W S
/
_output_shapes
:?????????

`
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_95189

inputs1
matmul_readvariableop_resource:	?K-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?K*
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
_construction_contextkEagerRuntime*+
_input_shapes
:??????????K: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????K
 
_user_specified_nameinputs
?

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_96307

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????  `C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????  `*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  `w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  `q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  `a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  `:W S
/
_output_shapes
:?????????  `
 
_user_specified_nameinputs
?
?
&__inference_conv2d_layer_call_fn_96088

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
A__inference_conv2d_layer_call_and_return_conditional_losses_94759w
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
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96262

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
?,
?
@__inference_model_layer_call_and_return_conditional_losses_95122
input_3&
conv2d_95088:@
conv2d_95090:@'
batch_normalization_95093:@'
batch_normalization_95095:@'
batch_normalization_95097:@'
batch_normalization_95099:@(
conv2d_1_95104:@`
conv2d_1_95106:`)
batch_normalization_1_95109:`)
batch_normalization_1_95111:`)
batch_normalization_1_95113:`)
batch_normalization_1_95115:`
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_95088conv2d_95090*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_94759?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_95093batch_normalization_95095batch_normalization_95097batch_normalization_95099*
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_94642?
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_94911?
max_pooling2d/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_94662?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_95104conv2d_1_95106*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_94793?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_95109batch_normalization_1_95111batch_normalization_1_95113batch_normalization_1_95115*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94718?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_94878?
max_pooling2d_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94738?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_94822p
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????K?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
??
?
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95686
inputs_0
inputs_1E
+model_conv2d_conv2d_readvariableop_resource:@:
,model_conv2d_biasadd_readvariableop_resource:@?
1model_batch_normalization_readvariableop_resource:@A
3model_batch_normalization_readvariableop_1_resource:@P
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource:@R
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_1_conv2d_readvariableop_resource:@`<
.model_conv2d_1_biasadd_readvariableop_resource:`A
3model_batch_normalization_1_readvariableop_resource:`C
5model_batch_normalization_1_readvariableop_1_resource:`R
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:`T
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:`7
$dense_matmul_readvariableop_resource:	?K3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?(model/batch_normalization/ReadVariableOp?*model/batch_normalization/ReadVariableOp_1?*model/batch_normalization/ReadVariableOp_2?*model/batch_normalization/ReadVariableOp_3?;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?=model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp??model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1?*model/batch_normalization_1/ReadVariableOp?,model/batch_normalization_1/ReadVariableOp_1?,model/batch_normalization_1/ReadVariableOp_2?,model/batch_normalization_1/ReadVariableOp_3?#model/conv2d/BiasAdd/ReadVariableOp?%model/conv2d/BiasAdd_1/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?$model/conv2d/Conv2D_1/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?'model/conv2d_1/BiasAdd_1/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?&model/conv2d_1/Conv2D_1/ReadVariableOp?
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
model/dropout/IdentityIdentity.model/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????mm@?
model/max_pooling2d/MaxPoolMaxPoolmodel/dropout/Identity:output:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0?
model/conv2d_1/Conv2DConv2D$model/max_pooling2d/MaxPool:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `*
paddingVALID*
strides
?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `v
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  `?
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:`*
dtype0?
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!model/conv2d_1/Relu:activations:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  `:`:`:`:`:*
epsilon%o?:*
is_training( ?
model/dropout_1/IdentityIdentity0model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  `?
model/max_pooling2d_1/MaxPoolMaxPool!model/dropout_1/Identity:output:0*/
_output_shapes
:?????????

`*
ksize
*
paddingVALID*
strides
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????%  ?
model/flatten/ReshapeReshape&model/max_pooling2d_1/MaxPool:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????K?
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
model/dropout/Identity_1Identity0model/batch_normalization/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:?????????mm@?
model/max_pooling2d/MaxPool_1MaxPool!model/dropout/Identity_1:output:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
?
&model/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0?
model/conv2d_1/Conv2D_1Conv2D&model/max_pooling2d/MaxPool_1:output:0.model/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `*
paddingVALID*
strides
?
'model/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0?
model/conv2d_1/BiasAdd_1BiasAdd model/conv2d_1/Conv2D_1:output:0/model/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `z
model/conv2d_1/Relu_1Relu!model/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????  `?
,model/batch_normalization_1/ReadVariableOp_2ReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:`*
dtype0?
,model/batch_normalization_1/ReadVariableOp_3ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
=model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0?
?model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
.model/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3#model/conv2d_1/Relu_1:activations:04model/batch_normalization_1/ReadVariableOp_2:value:04model/batch_normalization_1/ReadVariableOp_3:value:0Emodel/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0Gmodel/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  `:`:`:`:`:*
epsilon%o?:*
is_training( ?
model/dropout_1/Identity_1Identity2model/batch_normalization_1/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:?????????  `?
model/max_pooling2d_1/MaxPool_1MaxPool#model/dropout_1/Identity_1:output:0*/
_output_shapes
:?????????

`*
ksize
*
paddingVALID*
strides
f
model/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"?????%  ?
model/flatten/Reshape_1Reshape(model/max_pooling2d_1/MaxPool_1:output:0model/flatten/Const_1:output:0*
T0*(
_output_shapes
:??????????K?
tf.math.subtract/SubSubmodel/flatten/Reshape:output:0 model/flatten/Reshape_1:output:0*
T0*(
_output_shapes
:??????????Kc
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*(
_output_shapes
:??????????Kf
dropout_2/IdentityIdentitytf.math.abs/Abs:y:0*
T0*(
_output_shapes
:??????????K?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?K*
dtype0?
dense/MatMulMatMuldropout_2/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
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
:??????????

NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1<^model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp>^model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1+^model/batch_normalization/ReadVariableOp_2+^model/batch_normalization/ReadVariableOp_3<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1>^model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp@^model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1-^model/batch_normalization_1/ReadVariableOp_2-^model/batch_normalization_1/ReadVariableOp_3$^model/conv2d/BiasAdd/ReadVariableOp&^model/conv2d/BiasAdd_1/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp%^model/conv2d/Conv2D_1/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp(^model/conv2d_1/BiasAdd_1/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp'^model/conv2d_1/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12z
;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2~
=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12X
*model/batch_normalization/ReadVariableOp_2*model/batch_normalization/ReadVariableOp_22X
*model/batch_normalization/ReadVariableOp_3*model/batch_normalization/ReadVariableOp_32z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12~
=model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp=model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2?
?model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1?model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12\
,model/batch_normalization_1/ReadVariableOp_2,model/batch_normalization_1/ReadVariableOp_22\
,model/batch_normalization_1/ReadVariableOp_3,model/batch_normalization_1/ReadVariableOp_32J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2N
%model/conv2d/BiasAdd_1/ReadVariableOp%model/conv2d/BiasAdd_1/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2L
$model/conv2d/Conv2D_1/ReadVariableOp$model/conv2d/Conv2D_1/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2R
'model/conv2d_1/BiasAdd_1/ReadVariableOp'model/conv2d_1/BiasAdd_1/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2P
&model/conv2d_1/Conv2D_1/ReadVariableOp&model/conv2d_1/Conv2D_1/ReadVariableOp:[ W
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
?Z
?
@__inference_model_layer_call_and_return_conditional_losses_96032

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_1_conv2d_readvariableop_resource:@`6
(conv2d_1_biasadd_readvariableop_resource:`;
-batch_normalization_1_readvariableop_resource:`=
/batch_normalization_1_readvariableop_1_resource:`L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:`N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:`
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
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
dtype0Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout/dropout/MulMul(batch_normalization/FusedBatchNormV3:y:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????mm@m
dropout/dropout/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????mm@*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????mm@?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????mm@?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????mm@?
max_pooling2d/MaxPoolMaxPooldropout/dropout/Mul_1:z:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  `?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:`*
dtype0?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  `:`:`:`:`:*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/MulMul*batch_normalization_1/FusedBatchNormV3:y:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  `q
dropout_1/dropout/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  `*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  `?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  `?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  `?
max_pooling2d_1/MaxPoolMaxPooldropout_1/dropout/Mul_1:z:0*/
_output_shapes
:?????????

`*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????%  ?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????Kh
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*(
_output_shapes
:??????????K?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_94793

inputs8
conv2d_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  `i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  `w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????$$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????$$@
 
_user_specified_nameinputs
??
?
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95820
inputs_0
inputs_1E
+model_conv2d_conv2d_readvariableop_resource:@:
,model_conv2d_biasadd_readvariableop_resource:@?
1model_batch_normalization_readvariableop_resource:@A
3model_batch_normalization_readvariableop_1_resource:@P
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource:@R
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_1_conv2d_readvariableop_resource:@`<
.model_conv2d_1_biasadd_readvariableop_resource:`A
3model_batch_normalization_1_readvariableop_resource:`C
5model_batch_normalization_1_readvariableop_1_resource:`R
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:`T
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:`7
$dense_matmul_readvariableop_resource:	?K3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?(model/batch_normalization/AssignNewValue?*model/batch_normalization/AssignNewValue_1?*model/batch_normalization/AssignNewValue_2?*model/batch_normalization/AssignNewValue_3?9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?;model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?=model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?(model/batch_normalization/ReadVariableOp?*model/batch_normalization/ReadVariableOp_1?*model/batch_normalization/ReadVariableOp_2?*model/batch_normalization/ReadVariableOp_3?*model/batch_normalization_1/AssignNewValue?,model/batch_normalization_1/AssignNewValue_1?,model/batch_normalization_1/AssignNewValue_2?,model/batch_normalization_1/AssignNewValue_3?;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?=model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp??model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1?*model/batch_normalization_1/ReadVariableOp?,model/batch_normalization_1/ReadVariableOp_1?,model/batch_normalization_1/ReadVariableOp_2?,model/batch_normalization_1/ReadVariableOp_3?#model/conv2d/BiasAdd/ReadVariableOp?%model/conv2d/BiasAdd_1/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?$model/conv2d/Conv2D_1/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?'model/conv2d_1/BiasAdd_1/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?&model/conv2d_1/Conv2D_1/ReadVariableOp?
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
dtype0`
model/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
model/dropout/dropout/MulMul.model/batch_normalization/FusedBatchNormV3:y:0$model/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????mm@y
model/dropout/dropout/ShapeShape.model/batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:?
2model/dropout/dropout/random_uniform/RandomUniformRandomUniform$model/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????mm@*
dtype0i
$model/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"model/dropout/dropout/GreaterEqualGreaterEqual;model/dropout/dropout/random_uniform/RandomUniform:output:0-model/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????mm@?
model/dropout/dropout/CastCast&model/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????mm@?
model/dropout/dropout/Mul_1Mulmodel/dropout/dropout/Mul:z:0model/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????mm@?
model/max_pooling2d/MaxPoolMaxPoolmodel/dropout/dropout/Mul_1:z:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0?
model/conv2d_1/Conv2DConv2D$model/max_pooling2d/MaxPool:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `*
paddingVALID*
strides
?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `v
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  `?
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:`*
dtype0?
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!model/conv2d_1/Relu:activations:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  `:`:`:`:`:*
epsilon%o?:*
exponential_avg_factor%
?#<?
*model/batch_normalization_1/AssignNewValueAssignVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource9model/batch_normalization_1/FusedBatchNormV3:batch_mean:0<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
,model/batch_normalization_1/AssignNewValue_1AssignVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource=model/batch_normalization_1/FusedBatchNormV3:batch_variance:0>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0b
model/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
model/dropout_1/dropout/MulMul0model/batch_normalization_1/FusedBatchNormV3:y:0&model/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  `}
model/dropout_1/dropout/ShapeShape0model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:?
4model/dropout_1/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  `*
dtype0k
&model/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
$model/dropout_1/dropout/GreaterEqualGreaterEqual=model/dropout_1/dropout/random_uniform/RandomUniform:output:0/model/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  `?
model/dropout_1/dropout/CastCast(model/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  `?
model/dropout_1/dropout/Mul_1Mulmodel/dropout_1/dropout/Mul:z:0 model/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  `?
model/max_pooling2d_1/MaxPoolMaxPool!model/dropout_1/dropout/Mul_1:z:0*/
_output_shapes
:?????????

`*
ksize
*
paddingVALID*
strides
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????%  ?
model/flatten/ReshapeReshape&model/max_pooling2d_1/MaxPool:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????K?
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
dtype0b
model/dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
model/dropout/dropout_1/MulMul0model/batch_normalization/FusedBatchNormV3_1:y:0&model/dropout/dropout_1/Const:output:0*
T0*/
_output_shapes
:?????????mm@}
model/dropout/dropout_1/ShapeShape0model/batch_normalization/FusedBatchNormV3_1:y:0*
T0*
_output_shapes
:?
4model/dropout/dropout_1/random_uniform/RandomUniformRandomUniform&model/dropout/dropout_1/Shape:output:0*
T0*/
_output_shapes
:?????????mm@*
dtype0k
&model/dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
$model/dropout/dropout_1/GreaterEqualGreaterEqual=model/dropout/dropout_1/random_uniform/RandomUniform:output:0/model/dropout/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????mm@?
model/dropout/dropout_1/CastCast(model/dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????mm@?
model/dropout/dropout_1/Mul_1Mulmodel/dropout/dropout_1/Mul:z:0 model/dropout/dropout_1/Cast:y:0*
T0*/
_output_shapes
:?????????mm@?
model/max_pooling2d/MaxPool_1MaxPool!model/dropout/dropout_1/Mul_1:z:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
?
&model/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0?
model/conv2d_1/Conv2D_1Conv2D&model/max_pooling2d/MaxPool_1:output:0.model/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `*
paddingVALID*
strides
?
'model/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0?
model/conv2d_1/BiasAdd_1BiasAdd model/conv2d_1/Conv2D_1:output:0/model/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `z
model/conv2d_1/Relu_1Relu!model/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????  `?
,model/batch_normalization_1/ReadVariableOp_2ReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:`*
dtype0?
,model/batch_normalization_1/ReadVariableOp_3ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
=model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource+^model/batch_normalization_1/AssignNewValue*
_output_shapes
:`*
dtype0?
?model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource-^model/batch_normalization_1/AssignNewValue_1*
_output_shapes
:`*
dtype0?
.model/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3#model/conv2d_1/Relu_1:activations:04model/batch_normalization_1/ReadVariableOp_2:value:04model/batch_normalization_1/ReadVariableOp_3:value:0Emodel/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0Gmodel/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  `:`:`:`:`:*
epsilon%o?:*
exponential_avg_factor%
?#<?
,model/batch_normalization_1/AssignNewValue_2AssignVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource;model/batch_normalization_1/FusedBatchNormV3_1:batch_mean:0+^model/batch_normalization_1/AssignNewValue>^model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0?
,model/batch_normalization_1/AssignNewValue_3AssignVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource?model/batch_normalization_1/FusedBatchNormV3_1:batch_variance:0-^model/batch_normalization_1/AssignNewValue_1@^model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0d
model/dropout_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
model/dropout_1/dropout_1/MulMul2model/batch_normalization_1/FusedBatchNormV3_1:y:0(model/dropout_1/dropout_1/Const:output:0*
T0*/
_output_shapes
:?????????  `?
model/dropout_1/dropout_1/ShapeShape2model/batch_normalization_1/FusedBatchNormV3_1:y:0*
T0*
_output_shapes
:?
6model/dropout_1/dropout_1/random_uniform/RandomUniformRandomUniform(model/dropout_1/dropout_1/Shape:output:0*
T0*/
_output_shapes
:?????????  `*
dtype0m
(model/dropout_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&model/dropout_1/dropout_1/GreaterEqualGreaterEqual?model/dropout_1/dropout_1/random_uniform/RandomUniform:output:01model/dropout_1/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  `?
model/dropout_1/dropout_1/CastCast*model/dropout_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  `?
model/dropout_1/dropout_1/Mul_1Mul!model/dropout_1/dropout_1/Mul:z:0"model/dropout_1/dropout_1/Cast:y:0*
T0*/
_output_shapes
:?????????  `?
model/max_pooling2d_1/MaxPool_1MaxPool#model/dropout_1/dropout_1/Mul_1:z:0*/
_output_shapes
:?????????

`*
ksize
*
paddingVALID*
strides
f
model/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"?????%  ?
model/flatten/Reshape_1Reshape(model/max_pooling2d_1/MaxPool_1:output:0model/flatten/Const_1:output:0*
T0*(
_output_shapes
:??????????K?
tf.math.subtract/SubSubmodel/flatten/Reshape:output:0 model/flatten/Reshape_1:output:0*
T0*(
_output_shapes
:??????????Kc
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*(
_output_shapes
:??????????K\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_2/dropout/MulMultf.math.abs/Abs:y:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????KZ
dropout_2/dropout/ShapeShapetf.math.abs/Abs:y:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????K*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????K?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????K?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????K?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?K*
dtype0?
dense/MatMulMatMuldropout_2/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
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
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp)^model/batch_normalization/AssignNewValue+^model/batch_normalization/AssignNewValue_1+^model/batch_normalization/AssignNewValue_2+^model/batch_normalization/AssignNewValue_3:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1<^model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp>^model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1+^model/batch_normalization/ReadVariableOp_2+^model/batch_normalization/ReadVariableOp_3+^model/batch_normalization_1/AssignNewValue-^model/batch_normalization_1/AssignNewValue_1-^model/batch_normalization_1/AssignNewValue_2-^model/batch_normalization_1/AssignNewValue_3<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1>^model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp@^model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1-^model/batch_normalization_1/ReadVariableOp_2-^model/batch_normalization_1/ReadVariableOp_3$^model/conv2d/BiasAdd/ReadVariableOp&^model/conv2d/BiasAdd_1/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp%^model/conv2d/Conv2D_1/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp(^model/conv2d_1/BiasAdd_1/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp'^model/conv2d_1/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 2<
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
*model/batch_normalization/ReadVariableOp_3*model/batch_normalization/ReadVariableOp_32X
*model/batch_normalization_1/AssignNewValue*model/batch_normalization_1/AssignNewValue2\
,model/batch_normalization_1/AssignNewValue_1,model/batch_normalization_1/AssignNewValue_12\
,model/batch_normalization_1/AssignNewValue_2,model/batch_normalization_1/AssignNewValue_22\
,model/batch_normalization_1/AssignNewValue_3,model/batch_normalization_1/AssignNewValue_32z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12~
=model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp=model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2?
?model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1?model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12\
,model/batch_normalization_1/ReadVariableOp_2,model/batch_normalization_1/ReadVariableOp_22\
,model/batch_normalization_1/ReadVariableOp_3,model/batch_normalization_1/ReadVariableOp_32J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2N
%model/conv2d/BiasAdd_1/ReadVariableOp%model/conv2d/BiasAdd_1/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2L
$model/conv2d/Conv2D_1/ReadVariableOp$model/conv2d/Conv2D_1/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2R
'model/conv2d_1/BiasAdd_1/ReadVariableOp'model/conv2d_1/BiasAdd_1/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2P
&model/conv2d_1/Conv2D_1/ReadVariableOp&model/conv2d_1/Conv2D_1/ReadVariableOp:[ W
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
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_96295

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  `c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????  `"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  `:W S
/
_output_shapes
:?????????  `
 
_user_specified_nameinputs
?,
?
@__inference_model_layer_call_and_return_conditional_losses_94992

inputs&
conv2d_94958:@
conv2d_94960:@'
batch_normalization_94963:@'
batch_normalization_94965:@'
batch_normalization_94967:@'
batch_normalization_94969:@(
conv2d_1_94974:@`
conv2d_1_94976:`)
batch_normalization_1_94979:`)
batch_normalization_1_94981:`)
batch_normalization_1_94983:`)
batch_normalization_1_94985:`
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_94958conv2d_94960*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_94759?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_94963batch_normalization_94965batch_normalization_94967batch_normalization_94969*
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_94642?
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_94911?
max_pooling2d/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_94662?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_94974conv2d_1_94976*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_94793?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_94979batch_normalization_1_94981batch_normalization_1_94983batch_normalization_1_94985*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94718?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_94878?
max_pooling2d_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94738?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_94822p
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????K?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95348

inputs
inputs_1%
model_95301:@
model_95303:@
model_95305:@
model_95307:@
model_95309:@
model_95311:@%
model_95313:@`
model_95315:`
model_95317:`
model_95319:`
model_95321:`
model_95323:`
dense_95342:	?K
dense_95344:
identity??dense/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_95301model_95303model_95305model_95307model_95309model_95311model_95313model_95315model_95317model_95319model_95321model_95323*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94992?
model/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_95301model_95303model_95305model_95307model_95309model_95311model_95313model_95315model_95317model_95319model_95321model_95323^model/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94992?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:??????????Kc
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*(
_output_shapes
:??????????K?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCalltf.math.abs/Abs:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_95257?
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_95342dense_95344*
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
@__inference_dense_layer_call_and_return_conditional_losses_95189u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2>
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
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94718

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
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
-:+???????????????????????????`?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_96079

inputs1
matmul_readvariableop_resource:	?K-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?K*
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
_construction_contextkEagerRuntime*+
_input_shapes
:??????????K: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????K
 
_user_specified_nameinputs
?Z
?
__inference__traced_save_96490
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop$
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
:savev2_adam_batch_normalization_beta_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*?
value?B?/B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/2/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop/savev2_weights_intermediate_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/	?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?K:: : : : :@:@:@:@:@:@:@`:`:`:`:`:`: : : : : : : : :	?K::@:@:@:@:@`:`:`:`:	?K::@:@:@:@:@`:`:`:`: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?K: 
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
: :,(
&
_output_shapes
:@: 

_output_shapes
:@: 	
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
:@:,(
&
_output_shapes
:@`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?K: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:  

_output_shapes
:@:,!(
&
_output_shapes
:@`: "

_output_shapes
:`: #

_output_shapes
:`: $

_output_shapes
:`:%%!

_output_shapes
:	?K: &

_output_shapes
::,'(
&
_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@:,+(
&
_output_shapes
:@`: ,

_output_shapes
:`: -

_output_shapes
:`: .

_output_shapes
:`:/

_output_shapes
: 
?
?
0__inference_model_3_rescaled_layer_call_fn_95587
inputs_0
inputs_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`

unknown_11:	?K

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95348o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 22
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
?
?
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95464
input_1
input_2%
model_95417:@
model_95419:@
model_95421:@
model_95423:@
model_95425:@
model_95427:@%
model_95429:@`
model_95431:`
model_95433:`
model_95435:`
model_95437:`
model_95439:`
dense_95458:	?K
dense_95460:
identity??dense/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_95417model_95419model_95421model_95423model_95425model_95427model_95429model_95431model_95433model_95435model_95437model_95439*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94825?
model/StatefulPartitionedCall_1StatefulPartitionedCallinput_2model_95417model_95419model_95421model_95423model_95425model_95427model_95429model_95431model_95433model_95435model_95437model_95439*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94825?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:??????????Kc
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*(
_output_shapes
:??????????K?
dropout_2/PartitionedCallPartitionedCalltf.math.abs/Abs:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_95176?
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_95458dense_95460*
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
@__inference_dense_layer_call_and_return_conditional_losses_95189u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 2>
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
?
A__inference_conv2d_layer_call_and_return_conditional_losses_94759

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
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_96198

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
%__inference_dense_layer_call_fn_96068

inputs
unknown:	?K
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
@__inference_dense_layer_call_and_return_conditional_losses_95189o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????K: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????K
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_96285

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_94813h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  `:W S
/
_output_shapes
:?????????  `
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_96143

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
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_96047

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????K\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????K"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????K:P L
(
_output_shapes
:??????????K
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_1_layer_call_fn_96312

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
GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94738?
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
˪
?
 __inference__wrapped_model_94589
input_1
input_2V
<model_3_rescaled_model_conv2d_conv2d_readvariableop_resource:@K
=model_3_rescaled_model_conv2d_biasadd_readvariableop_resource:@P
Bmodel_3_rescaled_model_batch_normalization_readvariableop_resource:@R
Dmodel_3_rescaled_model_batch_normalization_readvariableop_1_resource:@a
Smodel_3_rescaled_model_batch_normalization_fusedbatchnormv3_readvariableop_resource:@c
Umodel_3_rescaled_model_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@X
>model_3_rescaled_model_conv2d_1_conv2d_readvariableop_resource:@`M
?model_3_rescaled_model_conv2d_1_biasadd_readvariableop_resource:`R
Dmodel_3_rescaled_model_batch_normalization_1_readvariableop_resource:`T
Fmodel_3_rescaled_model_batch_normalization_1_readvariableop_1_resource:`c
Umodel_3_rescaled_model_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:`e
Wmodel_3_rescaled_model_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:`H
5model_3_rescaled_dense_matmul_readvariableop_resource:	?KD
6model_3_rescaled_dense_biasadd_readvariableop_resource:
identity??-model_3_rescaled/dense/BiasAdd/ReadVariableOp?,model_3_rescaled/dense/MatMul/ReadVariableOp?Jmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOp?Lmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?Lmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?Nmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1?9model_3_rescaled/model/batch_normalization/ReadVariableOp?;model_3_rescaled/model/batch_normalization/ReadVariableOp_1?;model_3_rescaled/model/batch_normalization/ReadVariableOp_2?;model_3_rescaled/model/batch_normalization/ReadVariableOp_3?Lmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Nmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?Nmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp?Pmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1?;model_3_rescaled/model/batch_normalization_1/ReadVariableOp?=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_1?=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_2?=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_3?4model_3_rescaled/model/conv2d/BiasAdd/ReadVariableOp?6model_3_rescaled/model/conv2d/BiasAdd_1/ReadVariableOp?3model_3_rescaled/model/conv2d/Conv2D/ReadVariableOp?5model_3_rescaled/model/conv2d/Conv2D_1/ReadVariableOp?6model_3_rescaled/model/conv2d_1/BiasAdd/ReadVariableOp?8model_3_rescaled/model/conv2d_1/BiasAdd_1/ReadVariableOp?5model_3_rescaled/model/conv2d_1/Conv2D/ReadVariableOp?7model_3_rescaled/model/conv2d_1/Conv2D_1/ReadVariableOp?
3model_3_rescaled/model/conv2d/Conv2D/ReadVariableOpReadVariableOp<model_3_rescaled_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
$model_3_rescaled/model/conv2d/Conv2DConv2Dinput_1;model_3_rescaled/model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
?
4model_3_rescaled/model/conv2d/BiasAdd/ReadVariableOpReadVariableOp=model_3_rescaled_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
%model_3_rescaled/model/conv2d/BiasAddBiasAdd-model_3_rescaled/model/conv2d/Conv2D:output:0<model_3_rescaled/model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@?
"model_3_rescaled/model/conv2d/ReluRelu.model_3_rescaled/model/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????mm@?
9model_3_rescaled/model/batch_normalization/ReadVariableOpReadVariableOpBmodel_3_rescaled_model_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0?
;model_3_rescaled/model/batch_normalization/ReadVariableOp_1ReadVariableOpDmodel_3_rescaled_model_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Jmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSmodel_3_rescaled_model_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Lmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpUmodel_3_rescaled_model_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
;model_3_rescaled/model/batch_normalization/FusedBatchNormV3FusedBatchNormV30model_3_rescaled/model/conv2d/Relu:activations:0Amodel_3_rescaled/model/batch_normalization/ReadVariableOp:value:0Cmodel_3_rescaled/model/batch_normalization/ReadVariableOp_1:value:0Rmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Tmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????mm@:@:@:@:@:*
epsilon%o?:*
is_training( ?
'model_3_rescaled/model/dropout/IdentityIdentity?model_3_rescaled/model/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????mm@?
,model_3_rescaled/model/max_pooling2d/MaxPoolMaxPool0model_3_rescaled/model/dropout/Identity:output:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
?
5model_3_rescaled/model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp>model_3_rescaled_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0?
&model_3_rescaled/model/conv2d_1/Conv2DConv2D5model_3_rescaled/model/max_pooling2d/MaxPool:output:0=model_3_rescaled/model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `*
paddingVALID*
strides
?
6model_3_rescaled/model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?model_3_rescaled_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0?
'model_3_rescaled/model/conv2d_1/BiasAddBiasAdd/model_3_rescaled/model/conv2d_1/Conv2D:output:0>model_3_rescaled/model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `?
$model_3_rescaled/model/conv2d_1/ReluRelu0model_3_rescaled/model/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  `?
;model_3_rescaled/model/batch_normalization_1/ReadVariableOpReadVariableOpDmodel_3_rescaled_model_batch_normalization_1_readvariableop_resource*
_output_shapes
:`*
dtype0?
=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_1ReadVariableOpFmodel_3_rescaled_model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
Lmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_3_rescaled_model_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0?
Nmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_3_rescaled_model_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
=model_3_rescaled/model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV32model_3_rescaled/model/conv2d_1/Relu:activations:0Cmodel_3_rescaled/model/batch_normalization_1/ReadVariableOp:value:0Emodel_3_rescaled/model/batch_normalization_1/ReadVariableOp_1:value:0Tmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Vmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  `:`:`:`:`:*
epsilon%o?:*
is_training( ?
)model_3_rescaled/model/dropout_1/IdentityIdentityAmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  `?
.model_3_rescaled/model/max_pooling2d_1/MaxPoolMaxPool2model_3_rescaled/model/dropout_1/Identity:output:0*/
_output_shapes
:?????????

`*
ksize
*
paddingVALID*
strides
u
$model_3_rescaled/model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????%  ?
&model_3_rescaled/model/flatten/ReshapeReshape7model_3_rescaled/model/max_pooling2d_1/MaxPool:output:0-model_3_rescaled/model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????K?
5model_3_rescaled/model/conv2d/Conv2D_1/ReadVariableOpReadVariableOp<model_3_rescaled_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
&model_3_rescaled/model/conv2d/Conv2D_1Conv2Dinput_2=model_3_rescaled/model/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@*
paddingVALID*
strides
?
6model_3_rescaled/model/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp=model_3_rescaled_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
'model_3_rescaled/model/conv2d/BiasAdd_1BiasAdd/model_3_rescaled/model/conv2d/Conv2D_1:output:0>model_3_rescaled/model/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm@?
$model_3_rescaled/model/conv2d/Relu_1Relu0model_3_rescaled/model/conv2d/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????mm@?
;model_3_rescaled/model/batch_normalization/ReadVariableOp_2ReadVariableOpBmodel_3_rescaled_model_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0?
;model_3_rescaled/model/batch_normalization/ReadVariableOp_3ReadVariableOpDmodel_3_rescaled_model_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Lmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpSmodel_3_rescaled_model_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Nmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpUmodel_3_rescaled_model_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
=model_3_rescaled/model/batch_normalization/FusedBatchNormV3_1FusedBatchNormV32model_3_rescaled/model/conv2d/Relu_1:activations:0Cmodel_3_rescaled/model/batch_normalization/ReadVariableOp_2:value:0Cmodel_3_rescaled/model/batch_normalization/ReadVariableOp_3:value:0Tmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Vmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????mm@:@:@:@:@:*
epsilon%o?:*
is_training( ?
)model_3_rescaled/model/dropout/Identity_1IdentityAmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:?????????mm@?
.model_3_rescaled/model/max_pooling2d/MaxPool_1MaxPool2model_3_rescaled/model/dropout/Identity_1:output:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
?
7model_3_rescaled/model/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp>model_3_rescaled_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0?
(model_3_rescaled/model/conv2d_1/Conv2D_1Conv2D7model_3_rescaled/model/max_pooling2d/MaxPool_1:output:0?model_3_rescaled/model/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `*
paddingVALID*
strides
?
8model_3_rescaled/model/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp?model_3_rescaled_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0?
)model_3_rescaled/model/conv2d_1/BiasAdd_1BiasAdd1model_3_rescaled/model/conv2d_1/Conv2D_1:output:0@model_3_rescaled/model/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `?
&model_3_rescaled/model/conv2d_1/Relu_1Relu2model_3_rescaled/model/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:?????????  `?
=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_2ReadVariableOpDmodel_3_rescaled_model_batch_normalization_1_readvariableop_resource*
_output_shapes
:`*
dtype0?
=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_3ReadVariableOpFmodel_3_rescaled_model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
Nmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpUmodel_3_rescaled_model_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0?
Pmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpWmodel_3_rescaled_model_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
?model_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV34model_3_rescaled/model/conv2d_1/Relu_1:activations:0Emodel_3_rescaled/model/batch_normalization_1/ReadVariableOp_2:value:0Emodel_3_rescaled/model/batch_normalization_1/ReadVariableOp_3:value:0Vmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0Xmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  `:`:`:`:`:*
epsilon%o?:*
is_training( ?
+model_3_rescaled/model/dropout_1/Identity_1IdentityCmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:?????????  `?
0model_3_rescaled/model/max_pooling2d_1/MaxPool_1MaxPool4model_3_rescaled/model/dropout_1/Identity_1:output:0*/
_output_shapes
:?????????

`*
ksize
*
paddingVALID*
strides
w
&model_3_rescaled/model/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"?????%  ?
(model_3_rescaled/model/flatten/Reshape_1Reshape9model_3_rescaled/model/max_pooling2d_1/MaxPool_1:output:0/model_3_rescaled/model/flatten/Const_1:output:0*
T0*(
_output_shapes
:??????????K?
%model_3_rescaled/tf.math.subtract/SubSub/model_3_rescaled/model/flatten/Reshape:output:01model_3_rescaled/model/flatten/Reshape_1:output:0*
T0*(
_output_shapes
:??????????K?
 model_3_rescaled/tf.math.abs/AbsAbs)model_3_rescaled/tf.math.subtract/Sub:z:0*
T0*(
_output_shapes
:??????????K?
#model_3_rescaled/dropout_2/IdentityIdentity$model_3_rescaled/tf.math.abs/Abs:y:0*
T0*(
_output_shapes
:??????????K?
,model_3_rescaled/dense/MatMul/ReadVariableOpReadVariableOp5model_3_rescaled_dense_matmul_readvariableop_resource*
_output_shapes
:	?K*
dtype0?
model_3_rescaled/dense/MatMulMatMul,model_3_rescaled/dropout_2/Identity:output:04model_3_rescaled/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-model_3_rescaled/dense/BiasAdd/ReadVariableOpReadVariableOp6model_3_rescaled_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_3_rescaled/dense/BiasAddBiasAdd'model_3_rescaled/dense/MatMul:product:05model_3_rescaled/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
model_3_rescaled/dense/SigmoidSigmoid'model_3_rescaled/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????q
IdentityIdentity"model_3_rescaled/dense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^model_3_rescaled/dense/BiasAdd/ReadVariableOp-^model_3_rescaled/dense/MatMul/ReadVariableOpK^model_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOpM^model_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1M^model_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOpO^model_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:^model_3_rescaled/model/batch_normalization/ReadVariableOp<^model_3_rescaled/model/batch_normalization/ReadVariableOp_1<^model_3_rescaled/model/batch_normalization/ReadVariableOp_2<^model_3_rescaled/model/batch_normalization/ReadVariableOp_3M^model_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpO^model_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1O^model_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpQ^model_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1<^model_3_rescaled/model/batch_normalization_1/ReadVariableOp>^model_3_rescaled/model/batch_normalization_1/ReadVariableOp_1>^model_3_rescaled/model/batch_normalization_1/ReadVariableOp_2>^model_3_rescaled/model/batch_normalization_1/ReadVariableOp_35^model_3_rescaled/model/conv2d/BiasAdd/ReadVariableOp7^model_3_rescaled/model/conv2d/BiasAdd_1/ReadVariableOp4^model_3_rescaled/model/conv2d/Conv2D/ReadVariableOp6^model_3_rescaled/model/conv2d/Conv2D_1/ReadVariableOp7^model_3_rescaled/model/conv2d_1/BiasAdd/ReadVariableOp9^model_3_rescaled/model/conv2d_1/BiasAdd_1/ReadVariableOp6^model_3_rescaled/model/conv2d_1/Conv2D/ReadVariableOp8^model_3_rescaled/model/conv2d_1/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 2^
-model_3_rescaled/dense/BiasAdd/ReadVariableOp-model_3_rescaled/dense/BiasAdd/ReadVariableOp2\
,model_3_rescaled/dense/MatMul/ReadVariableOp,model_3_rescaled/dense/MatMul/ReadVariableOp2?
Jmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOpJmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Lmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Lmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
Lmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOpLmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2?
Nmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1Nmodel_3_rescaled/model/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12v
9model_3_rescaled/model/batch_normalization/ReadVariableOp9model_3_rescaled/model/batch_normalization/ReadVariableOp2z
;model_3_rescaled/model/batch_normalization/ReadVariableOp_1;model_3_rescaled/model/batch_normalization/ReadVariableOp_12z
;model_3_rescaled/model/batch_normalization/ReadVariableOp_2;model_3_rescaled/model/batch_normalization/ReadVariableOp_22z
;model_3_rescaled/model/batch_normalization/ReadVariableOp_3;model_3_rescaled/model/batch_normalization/ReadVariableOp_32?
Lmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpLmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Nmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Nmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12?
Nmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpNmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2?
Pmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1Pmodel_3_rescaled/model/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12z
;model_3_rescaled/model/batch_normalization_1/ReadVariableOp;model_3_rescaled/model/batch_normalization_1/ReadVariableOp2~
=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_1=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_12~
=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_2=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_22~
=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_3=model_3_rescaled/model/batch_normalization_1/ReadVariableOp_32l
4model_3_rescaled/model/conv2d/BiasAdd/ReadVariableOp4model_3_rescaled/model/conv2d/BiasAdd/ReadVariableOp2p
6model_3_rescaled/model/conv2d/BiasAdd_1/ReadVariableOp6model_3_rescaled/model/conv2d/BiasAdd_1/ReadVariableOp2j
3model_3_rescaled/model/conv2d/Conv2D/ReadVariableOp3model_3_rescaled/model/conv2d/Conv2D/ReadVariableOp2n
5model_3_rescaled/model/conv2d/Conv2D_1/ReadVariableOp5model_3_rescaled/model/conv2d/Conv2D_1/ReadVariableOp2p
6model_3_rescaled/model/conv2d_1/BiasAdd/ReadVariableOp6model_3_rescaled/model/conv2d_1/BiasAdd/ReadVariableOp2t
8model_3_rescaled/model/conv2d_1/BiasAdd_1/ReadVariableOp8model_3_rescaled/model/conv2d_1/BiasAdd_1/ReadVariableOp2n
5model_3_rescaled/model/conv2d_1/Conv2D/ReadVariableOp5model_3_rescaled/model/conv2d_1/Conv2D/ReadVariableOp2r
7model_3_rescaled/model/conv2d_1/Conv2D_1/ReadVariableOp7model_3_rescaled/model/conv2d_1/Conv2D_1/ReadVariableOp:Z V
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
?
5__inference_batch_normalization_1_layer_call_fn_96231

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94687?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_96207

inputs!
unknown:@`
	unknown_0:`
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_94793w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????$$@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????$$@
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_1_layer_call_fn_96244

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94718?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_96176

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????mm@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????mm@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????mm@:W S
/
_output_shapes
:?????????mm@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94687

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
?

a
B__inference_dropout_layer_call_and_return_conditional_losses_96188

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????mm@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????mm@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????mm@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????mm@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????mm@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????mm@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????mm@:W S
/
_output_shapes
:?????????mm@
 
_user_specified_nameinputs
?)
?
@__inference_model_layer_call_and_return_conditional_losses_95085
input_3&
conv2d_95051:@
conv2d_95053:@'
batch_normalization_95056:@'
batch_normalization_95058:@'
batch_normalization_95060:@'
batch_normalization_95062:@(
conv2d_1_95067:@`
conv2d_1_95069:`)
batch_normalization_1_95072:`)
batch_normalization_1_95074:`)
batch_normalization_1_95076:`)
batch_normalization_1_95078:`
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_95051conv2d_95053*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_94759?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_95056batch_normalization_95058batch_normalization_95060batch_normalization_95062*
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_94611?
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_94779?
max_pooling2d/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_94662?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_95067conv2d_1_95069*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_94793?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_95072batch_normalization_1_95074batch_normalization_1_95076batch_normalization_1_95078*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94687?
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_94813?
max_pooling2d_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94738?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_94822p
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????K?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94738

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
?	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_96059

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????KC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????K*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????Kp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????Kj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????KZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????K"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????K:P L
(
_output_shapes
:??????????K
 
_user_specified_nameinputs
?
b
)__inference_dropout_2_layer_call_fn_96042

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_95257p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????K`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????K22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????K
 
_user_specified_nameinputs
?
I
-__inference_max_pooling2d_layer_call_fn_96193

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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_94662?
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
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_96328

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????%  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????KY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????K"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

`:W S
/
_output_shapes
:?????????

`
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_94611

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
?

?
%__inference_model_layer_call_fn_95914

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94992p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????K`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_95856
input_1
input_2!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`

unknown_11:	?K

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_94589o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 22
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
B__inference_flatten_layer_call_and_return_conditional_losses_94822

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????%  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????KY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????K"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

`:W S
/
_output_shapes
:?????????

`
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_95176

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????K\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????K"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????K:P L
(
_output_shapes
:??????????K
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_95885

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94825p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????K`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
0__inference_model_3_rescaled_layer_call_fn_95553
inputs_0
inputs_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`

unknown_11:	?K

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95196o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 22
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
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_94779

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????mm@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????mm@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????mm@:W S
/
_output_shapes
:?????????mm@
 
_user_specified_nameinputs
?
E
)__inference_dropout_2_layer_call_fn_96037

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_95176a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????K"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????K:P L
(
_output_shapes
:??????????K
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_96317

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
?	
?
3__inference_batch_normalization_layer_call_fn_96112

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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_94611?
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

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_94878

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????  `C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????  `*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  `w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  `q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  `a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  `:W S
/
_output_shapes
:?????????  `
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_96638
file_prefix0
assignvariableop_dense_kernel:	?K+
assignvariableop_1_dense_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: :
 assignvariableop_6_conv2d_kernel:@,
assignvariableop_7_conv2d_bias:@:
,assignvariableop_8_batch_normalization_gamma:@9
+assignvariableop_9_batch_normalization_beta:@A
3assignvariableop_10_batch_normalization_moving_mean:@E
7assignvariableop_11_batch_normalization_moving_variance:@=
#assignvariableop_12_conv2d_1_kernel:@`/
!assignvariableop_13_conv2d_1_bias:`=
/assignvariableop_14_batch_normalization_1_gamma:`<
.assignvariableop_15_batch_normalization_1_beta:`C
5assignvariableop_16_batch_normalization_1_moving_mean:`G
9assignvariableop_17_batch_normalization_1_moving_variance:`#
assignvariableop_18_total: #
assignvariableop_19_count: %
assignvariableop_20_total_1: %
assignvariableop_21_count_1: ,
"assignvariableop_22_true_positives: -
#assignvariableop_23_false_positives: -
#assignvariableop_24_false_negatives: 2
(assignvariableop_25_weights_intermediate: :
'assignvariableop_26_adam_dense_kernel_m:	?K3
%assignvariableop_27_adam_dense_bias_m:B
(assignvariableop_28_adam_conv2d_kernel_m:@4
&assignvariableop_29_adam_conv2d_bias_m:@B
4assignvariableop_30_adam_batch_normalization_gamma_m:@A
3assignvariableop_31_adam_batch_normalization_beta_m:@D
*assignvariableop_32_adam_conv2d_1_kernel_m:@`6
(assignvariableop_33_adam_conv2d_1_bias_m:`D
6assignvariableop_34_adam_batch_normalization_1_gamma_m:`C
5assignvariableop_35_adam_batch_normalization_1_beta_m:`:
'assignvariableop_36_adam_dense_kernel_v:	?K3
%assignvariableop_37_adam_dense_bias_v:B
(assignvariableop_38_adam_conv2d_kernel_v:@4
&assignvariableop_39_adam_conv2d_bias_v:@B
4assignvariableop_40_adam_batch_normalization_gamma_v:@A
3assignvariableop_41_adam_batch_normalization_beta_v:@D
*assignvariableop_42_adam_conv2d_1_kernel_v:@`6
(assignvariableop_43_adam_conv2d_1_bias_v:`D
6assignvariableop_44_adam_batch_normalization_1_gamma_v:`C
5assignvariableop_45_adam_batch_normalization_1_beta_v:`
identity_47??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*?
value?B?/B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/2/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	[
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
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2d_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_batch_normalization_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp+assignvariableop_9_batch_normalization_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp3assignvariableop_10_batch_normalization_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp7assignvariableop_11_batch_normalization_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_1_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_1_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_1_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_1_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_true_positivesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp#assignvariableop_23_false_positivesIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_false_negativesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_weights_intermediateIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_dense_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_conv2d_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_batch_normalization_gamma_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp3assignvariableop_31_adam_batch_normalization_beta_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_conv2d_1_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_conv2d_1_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_batch_normalization_1_gamma_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_batch_normalization_1_beta_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp%assignvariableop_37_adam_dense_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp&assignvariableop_39_adam_conv2d_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp4assignvariableop_40_adam_batch_normalization_gamma_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp3assignvariableop_41_adam_batch_normalization_beta_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_conv2d_1_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_conv2d_1_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_batch_normalization_1_gamma_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp5assignvariableop_45_adam_batch_normalization_1_beta_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_47IdentityIdentity_46:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_47Identity_47:output:0*q
_input_shapes`
^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
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
?

?
%__inference_model_layer_call_fn_94852
input_3!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94825p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????K`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?	
?
3__inference_batch_normalization_layer_call_fn_96125

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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_94642?
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
?)
?
@__inference_model_layer_call_and_return_conditional_losses_94825

inputs&
conv2d_94760:@
conv2d_94762:@'
batch_normalization_94765:@'
batch_normalization_94767:@'
batch_normalization_94769:@'
batch_normalization_94771:@(
conv2d_1_94794:@`
conv2d_1_94796:`)
batch_normalization_1_94799:`)
batch_normalization_1_94801:`)
batch_normalization_1_94803:`)
batch_normalization_1_94805:`
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_94760conv2d_94762*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_94759?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_94765batch_normalization_94767batch_normalization_94769batch_normalization_94771*
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_94611?
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_94779?
max_pooling2d/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_94662?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_94794conv2d_1_94796*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_94793?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_94799batch_normalization_1_94801batch_normalization_1_94803batch_normalization_1_94805*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94687?
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_94813?
max_pooling2d_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94738?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_94822p
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????K?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
0__inference_model_3_rescaled_layer_call_fn_95227
input_1
input_2!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`

unknown_11:	?K

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95196o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 22
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
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_94642

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
?

a
B__inference_dropout_layer_call_and_return_conditional_losses_94911

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????mm@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????mm@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????mm@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????mm@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????mm@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????mm@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????mm@:W S
/
_output_shapes
:?????????mm@
 
_user_specified_nameinputs
?	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_95257

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????KC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????K*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????Kp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????Kj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????KZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????K"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????K:P L
(
_output_shapes
:??????????K
 
_user_specified_nameinputs
?=
?

@__inference_model_layer_call_and_return_conditional_losses_95966

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_1_conv2d_readvariableop_resource:@`6
(conv2d_1_biasadd_readvariableop_resource:`;
-batch_normalization_1_readvariableop_resource:`=
/batch_normalization_1_readvariableop_1_resource:`L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:`N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:`
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
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
dropout/IdentityIdentity(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????mm@?
max_pooling2d/MaxPoolMaxPooldropout/Identity:output:0*/
_output_shapes
:?????????$$@*
ksize
*
paddingVALID*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  `j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  `?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:`*
dtype0?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  `:`:`:`:`:*
epsilon%o?:*
is_training( ?
dropout_1/IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  `?
max_pooling2d_1/MaxPoolMaxPooldropout_1/Identity:output:0*/
_output_shapes
:?????????

`*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????%  ?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????Kh
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*(
_output_shapes
:??????????K?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95515
input_1
input_2%
model_95468:@
model_95470:@
model_95472:@
model_95474:@
model_95476:@
model_95478:@%
model_95480:@`
model_95482:`
model_95484:`
model_95486:`
model_95488:`
model_95490:`
dense_95509:	?K
dense_95511:
identity??dense/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_95468model_95470model_95472model_95474model_95476model_95478model_95480model_95482model_95484model_95486model_95488model_95490*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94992?
model/StatefulPartitionedCall_1StatefulPartitionedCallinput_2model_95468model_95470model_95472model_95474model_95476model_95478model_95480model_95482model_95484model_95486model_95488model_95490^model/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94992?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:??????????Kc
tf.math.abs/AbsAbstf.math.subtract/Sub:z:0*
T0*(
_output_shapes
:??????????K?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCalltf.math.abs/Abs:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_95257?
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_95509dense_95511*
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
@__inference_dense_layer_call_and_return_conditional_losses_95189u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2>
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
C
'__inference_dropout_layer_call_fn_96166

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_94779h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????mm@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????mm@:W S
/
_output_shapes
:?????????mm@
 
_user_specified_nameinputs
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_96099

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
?

?
%__inference_model_layer_call_fn_95048
input_3!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????K**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_94992p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????K`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
?
0__inference_model_3_rescaled_layer_call_fn_95413
input_1
input_2!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`

unknown_11:	?K

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95348o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:???????????:???????????: : : : : : : : : : : : : : 22
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
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_94662

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
layer-5
layer_with_weights-1
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
layer-8
layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_network
(
!	keras_api"
_tf_keras_layer
(
"	keras_api"
_tf_keras_layer
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'_random_generator
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
?
2iter

3beta_1

4beta_2
	5decay*m?+m?6m?7m?8m?9m?<m?=m?>m??m?*v?+v?6v?7v?8v?9v?<v?=v?>v??v?"
	optimizer
?
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
*12
+13"
trackable_list_wrapper
f
60
71
82
93
<4
=5
>6
?7
*8
+9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_model_3_rescaled_layer_call_fn_95227
0__inference_model_3_rescaled_layer_call_fn_95553
0__inference_model_3_rescaled_layer_call_fn_95587
0__inference_model_3_rescaled_layer_call_fn_95413?
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
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95686
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95820
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95464
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95515?
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
 __inference__wrapped_model_94589input_1input_2"?
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
Gserving_default"
signature_map
"
_tf_keras_input_layer
?

6kernel
7bias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Naxis
	8gamma
9beta
:moving_mean
;moving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y_random_generator
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
?
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
?

<kernel
=bias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
?
haxis
	>gamma
?beta
@moving_mean
Amoving_variance
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
?
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s_random_generator
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
?
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
v
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11"
trackable_list_wrapper
X
60
71
82
93
<4
=5
>6
?7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_model_layer_call_fn_94852
%__inference_model_layer_call_fn_95885
%__inference_model_layer_call_fn_95914
%__inference_model_layer_call_fn_95048?
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
@__inference_model_layer_call_and_return_conditional_losses_95966
@__inference_model_layer_call_and_return_conditional_losses_96032
@__inference_model_layer_call_and_return_conditional_losses_95085
@__inference_model_layer_call_and_return_conditional_losses_95122?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
#	variables
$trainable_variables
%regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
)__inference_dropout_2_layer_call_fn_96037
)__inference_dropout_2_layer_call_fn_96042?
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
D__inference_dropout_2_layer_call_and_return_conditional_losses_96047
D__inference_dropout_2_layer_call_and_return_conditional_losses_96059?
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
:	?K2dense/kernel
:2
dense/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_dense_layer_call_fn_96068?
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
@__inference_dense_layer_call_and_return_conditional_losses_96079?
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
':%@2conv2d/kernel
:@2conv2d/bias
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
):'@`2conv2d_1/kernel
:`2conv2d_1/bias
):'`2batch_normalization_1/gamma
(:&`2batch_normalization_1/beta
1:/` (2!batch_normalization_1/moving_mean
5:3` (2%batch_normalization_1/moving_variance
<
:0
;1
@2
A3"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
#__inference_signature_wrapper_95856input_1input_2"?
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
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_conv2d_layer_call_fn_96088?
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
A__inference_conv2d_layer_call_and_return_conditional_losses_96099?
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
80
91
:2
;3"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
?2?
3__inference_batch_normalization_layer_call_fn_96112
3__inference_batch_normalization_layer_call_fn_96125?
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_96143
N__inference_batch_normalization_layer_call_and_return_conditional_losses_96161?
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
'__inference_dropout_layer_call_fn_96166
'__inference_dropout_layer_call_fn_96171?
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
B__inference_dropout_layer_call_and_return_conditional_losses_96176
B__inference_dropout_layer_call_and_return_conditional_losses_96188?
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_max_pooling2d_layer_call_fn_96193?
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_96198?
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
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_conv2d_1_layer_call_fn_96207?
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_96218?
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
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
?2?
5__inference_batch_normalization_1_layer_call_fn_96231
5__inference_batch_normalization_1_layer_call_fn_96244?
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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96262
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96280?
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
)__inference_dropout_1_layer_call_fn_96285
)__inference_dropout_1_layer_call_fn_96290?
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
D__inference_dropout_1_layer_call_and_return_conditional_losses_96295
D__inference_dropout_1_layer_call_and_return_conditional_losses_96307?
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_max_pooling2d_1_layer_call_fn_96312?
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_96317?
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_flatten_layer_call_fn_96322?
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
B__inference_flatten_layer_call_and_return_conditional_losses_96328?
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
<
:0
;1
@2
A3"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
?
?
init_shape
?true_positives
?false_positives
?false_negatives
?weights_intermediate
?	variables
?	keras_api"
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
:0
;1"
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
@0
A1"
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
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
:  (2true_positives
:  (2false_positives
:  (2false_negatives
 :  (2weights_intermediate
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
$:"	?K2Adam/dense/kernel/m
:2Adam/dense/bias/m
,:*@2Adam/conv2d/kernel/m
:@2Adam/conv2d/bias/m
,:*@2 Adam/batch_normalization/gamma/m
+:)@2Adam/batch_normalization/beta/m
.:,@`2Adam/conv2d_1/kernel/m
 :`2Adam/conv2d_1/bias/m
.:,`2"Adam/batch_normalization_1/gamma/m
-:+`2!Adam/batch_normalization_1/beta/m
$:"	?K2Adam/dense/kernel/v
:2Adam/dense/bias/v
,:*@2Adam/conv2d/kernel/v
:@2Adam/conv2d/bias/v
,:*@2 Adam/batch_normalization/gamma/v
+:)@2Adam/batch_normalization/beta/v
.:,@`2Adam/conv2d_1/kernel/v
 :`2Adam/conv2d_1/bias/v
.:,`2"Adam/batch_normalization_1/gamma/v
-:+`2!Adam/batch_normalization_1/beta/v?
 __inference__wrapped_model_94589?6789:;<=>?@A*+l?i
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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96262?>?@AM?J
C?@
:?7
inputs+???????????????????????????`
p 
? "??<
5?2
0+???????????????????????????`
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96280?>?@AM?J
C?@
:?7
inputs+???????????????????????????`
p
? "??<
5?2
0+???????????????????????????`
? ?
5__inference_batch_normalization_1_layer_call_fn_96231?>?@AM?J
C?@
:?7
inputs+???????????????????????????`
p 
? "2?/+???????????????????????????`?
5__inference_batch_normalization_1_layer_call_fn_96244?>?@AM?J
C?@
:?7
inputs+???????????????????????????`
p
? "2?/+???????????????????????????`?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_96143?89:;M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_96161?89:;M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
3__inference_batch_normalization_layer_call_fn_96112?89:;M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
3__inference_batch_normalization_layer_call_fn_96125?89:;M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_96218l<=7?4
-?*
(?%
inputs?????????$$@
? "-?*
#? 
0?????????  `
? ?
(__inference_conv2d_1_layer_call_fn_96207_<=7?4
-?*
(?%
inputs?????????$$@
? " ??????????  `?
A__inference_conv2d_layer_call_and_return_conditional_losses_96099n679?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????mm@
? ?
&__inference_conv2d_layer_call_fn_96088a679?6
/?,
*?'
inputs???????????
? " ??????????mm@?
@__inference_dense_layer_call_and_return_conditional_losses_96079]*+0?-
&?#
!?
inputs??????????K
? "%?"
?
0?????????
? y
%__inference_dense_layer_call_fn_96068P*+0?-
&?#
!?
inputs??????????K
? "???????????
D__inference_dropout_1_layer_call_and_return_conditional_losses_96295l;?8
1?.
(?%
inputs?????????  `
p 
? "-?*
#? 
0?????????  `
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_96307l;?8
1?.
(?%
inputs?????????  `
p
? "-?*
#? 
0?????????  `
? ?
)__inference_dropout_1_layer_call_fn_96285_;?8
1?.
(?%
inputs?????????  `
p 
? " ??????????  `?
)__inference_dropout_1_layer_call_fn_96290_;?8
1?.
(?%
inputs?????????  `
p
? " ??????????  `?
D__inference_dropout_2_layer_call_and_return_conditional_losses_96047^4?1
*?'
!?
inputs??????????K
p 
? "&?#
?
0??????????K
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_96059^4?1
*?'
!?
inputs??????????K
p
? "&?#
?
0??????????K
? ~
)__inference_dropout_2_layer_call_fn_96037Q4?1
*?'
!?
inputs??????????K
p 
? "???????????K~
)__inference_dropout_2_layer_call_fn_96042Q4?1
*?'
!?
inputs??????????K
p
? "???????????K?
B__inference_dropout_layer_call_and_return_conditional_losses_96176l;?8
1?.
(?%
inputs?????????mm@
p 
? "-?*
#? 
0?????????mm@
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_96188l;?8
1?.
(?%
inputs?????????mm@
p
? "-?*
#? 
0?????????mm@
? ?
'__inference_dropout_layer_call_fn_96166_;?8
1?.
(?%
inputs?????????mm@
p 
? " ??????????mm@?
'__inference_dropout_layer_call_fn_96171_;?8
1?.
(?%
inputs?????????mm@
p
? " ??????????mm@?
B__inference_flatten_layer_call_and_return_conditional_losses_96328a7?4
-?*
(?%
inputs?????????

`
? "&?#
?
0??????????K
? 
'__inference_flatten_layer_call_fn_96322T7?4
-?*
(?%
inputs?????????

`
? "???????????K?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_96317?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_96312?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_96198?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_96193?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95464?6789:;<=>?@A*+t?q
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
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95515?6789:;<=>?@A*+t?q
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
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95686?6789:;<=>?@A*+v?s
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
K__inference_model_3_rescaled_layer_call_and_return_conditional_losses_95820?6789:;<=>?@A*+v?s
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
0__inference_model_3_rescaled_layer_call_fn_95227?6789:;<=>?@A*+t?q
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
0__inference_model_3_rescaled_layer_call_fn_95413?6789:;<=>?@A*+t?q
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
0__inference_model_3_rescaled_layer_call_fn_95553?6789:;<=>?@A*+v?s
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
0__inference_model_3_rescaled_layer_call_fn_95587?6789:;<=>?@A*+v?s
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
@__inference_model_layer_call_and_return_conditional_losses_95085z6789:;<=>?@AB??
8?5
+?(
input_3???????????
p 

 
? "&?#
?
0??????????K
? ?
@__inference_model_layer_call_and_return_conditional_losses_95122z6789:;<=>?@AB??
8?5
+?(
input_3???????????
p

 
? "&?#
?
0??????????K
? ?
@__inference_model_layer_call_and_return_conditional_losses_95966y6789:;<=>?@AA?>
7?4
*?'
inputs???????????
p 

 
? "&?#
?
0??????????K
? ?
@__inference_model_layer_call_and_return_conditional_losses_96032y6789:;<=>?@AA?>
7?4
*?'
inputs???????????
p

 
? "&?#
?
0??????????K
? ?
%__inference_model_layer_call_fn_94852m6789:;<=>?@AB??
8?5
+?(
input_3???????????
p 

 
? "???????????K?
%__inference_model_layer_call_fn_95048m6789:;<=>?@AB??
8?5
+?(
input_3???????????
p

 
? "???????????K?
%__inference_model_layer_call_fn_95885l6789:;<=>?@AA?>
7?4
*?'
inputs???????????
p 

 
? "???????????K?
%__inference_model_layer_call_fn_95914l6789:;<=>?@AA?>
7?4
*?'
inputs???????????
p

 
? "???????????K?
#__inference_signature_wrapper_95856?6789:;<=>?@A*+}?z
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