
é¿
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

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
ú
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
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
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

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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-0-g3f878cff5b68öñ
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	K*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
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

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:@*
dtype0

batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_10/gamma

0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:@*
dtype0

batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_10/beta

/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:@*
dtype0

"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_10/moving_mean

6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_10/moving_variance

:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:@*
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:@`*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:`*
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:`*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:`*
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:`*
dtype0
¤
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
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

Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes
:	K*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_10/kernel/m

+Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/m*&
_output_shapes
:@*
dtype0

Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_10/bias/m
{
)Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/m*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_10/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_10/gamma/m

7Adam/batch_normalization_10/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_10/gamma/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_10/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_10/beta/m

6Adam/batch_normalization_10/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_10/beta/m*
_output_shapes
:@*
dtype0

Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*(
shared_nameAdam/conv2d_11/kernel/m

+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*&
_output_shapes
:@`*
dtype0

Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*&
shared_nameAdam/conv2d_11/bias/m
{
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes
:`*
dtype0

#Adam/batch_normalization_11/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*4
shared_name%#Adam/batch_normalization_11/gamma/m

7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/m*
_output_shapes
:`*
dtype0

"Adam/batch_normalization_11/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/batch_normalization_11/beta/m

6Adam/batch_normalization_11/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/m*
_output_shapes
:`*
dtype0

Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes
:	K*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_10/kernel/v

+Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/v*&
_output_shapes
:@*
dtype0

Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_10/bias/v
{
)Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/v*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_10/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_10/gamma/v

7Adam/batch_normalization_10/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_10/gamma/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_10/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_10/beta/v

6Adam/batch_normalization_10/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_10/beta/v*
_output_shapes
:@*
dtype0

Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*(
shared_nameAdam/conv2d_11/kernel/v

+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*&
_output_shapes
:@`*
dtype0

Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*&
shared_nameAdam/conv2d_11/bias/v
{
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes
:`*
dtype0

#Adam/batch_normalization_11/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*4
shared_name%#Adam/batch_normalization_11/gamma/v

7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/v*
_output_shapes
:`*
dtype0

"Adam/batch_normalization_11/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/batch_normalization_11/beta/v

6Adam/batch_normalization_11/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/v*
_output_shapes
:`*
dtype0

NoOpNoOp
h
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ôg
valueÊgBÇg BÀg
Û
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
ú
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
¥
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'_random_generator
(__call__
*)&call_and_return_all_conditional_losses* 
¦

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*
ù
2iter

3beta_1

4beta_2
	5decay*mÑ+mÒ6mÓ7mÔ8mÕ9mÖ<m×=mØ>mÙ?mÚ*vÛ+vÜ6vÝ7vÞ8vß9và<vá=vâ>vã?vä*
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
°
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
¦

6kernel
7bias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses*
Õ
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
¥
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y_random_generator
Z__call__
*[&call_and_return_all_conditional_losses* 

\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
¦

<kernel
=bias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses*
Õ
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
¥
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s_random_generator
t__call__
*u&call_and_return_all_conditional_losses* 

v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 

|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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
PJ
VARIABLE_VALUEconv2d_10/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_10/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_10/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_10/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"batch_normalization_10/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_normalization_10/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_11/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_11/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_11/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_11/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_11/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_11/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
 
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

0
1
2*
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
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

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
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

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
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

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
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

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
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

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
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

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
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

Átotal

Âcount
Ã	variables
Ä	keras_api*
M

Åtotal

Æcount
Ç
_fn_kwargs
È	variables
É	keras_api*

Ê
init_shape
Ëtrue_positives
Ìfalse_positives
Ífalse_negatives
Îweights_intermediate
Ï	variables
Ð	keras_api*
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
Á0
Â1*

Ã	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Å0
Æ1*

È	variables*
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
Ë0
Ì1
Í2
Î3*

Ï	variables*
{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_10/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_10/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/batch_normalization_10/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_10/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_11/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_11/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/batch_normalization_11/gamma/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_11/beta/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_10/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_10/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/batch_normalization_10/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_10/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_11/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_11/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/batch_normalization_11/gamma/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_11/beta/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_16Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿàà

serving_default_input_17Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿàà
Ý
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_16serving_default_input_17conv2d_10/kernelconv2d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancedense_5/kerneldense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_110499
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
©
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp(weights_intermediate/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp+Adam/conv2d_10/kernel/m/Read/ReadVariableOp)Adam/conv2d_10/bias/m/Read/ReadVariableOp7Adam/batch_normalization_10/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_10/beta/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_11/beta/m/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp+Adam/conv2d_10/kernel/v/Read/ReadVariableOp)Adam/conv2d_10/bias/v/Read/ReadVariableOp7Adam/batch_normalization_10/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_10/beta/v/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_11/beta/v/Read/ReadVariableOpConst*;
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_111133

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayconv2d_10/kernelconv2d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancetotalcounttotal_1count_1true_positivesfalse_positivesfalse_negativesweights_intermediateAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/conv2d_10/kernel/mAdam/conv2d_10/bias/m#Adam/batch_normalization_10/gamma/m"Adam/batch_normalization_10/beta/mAdam/conv2d_11/kernel/mAdam/conv2d_11/bias/m#Adam/batch_normalization_11/gamma/m"Adam/batch_normalization_11/beta/mAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/conv2d_10/kernel/vAdam/conv2d_10/bias/v#Adam/batch_normalization_10/gamma/v"Adam/batch_normalization_10/beta/vAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/v#Adam/batch_normalization_11/gamma/v"Adam/batch_normalization_11/beta/v*:
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_111281÷

þ
E__inference_conv2d_11_layer_call_and_return_conditional_losses_110861

inputs8
conv2d_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
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
:ÿÿÿÿÿÿÿÿÿ  `X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ$$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@
 
_user_specified_nameinputs
Í

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_109254

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

d
+__inference_dropout_15_layer_call_fn_110814

inputs
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_109554w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿmm@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
 
_user_specified_nameinputs
ù
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_109456

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  `:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
 
_user_specified_nameinputs

d
+__inference_dropout_16_layer_call_fn_110933

inputs
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_109521w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  `22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
 
_user_specified_nameinputs
Æ

(__inference_dense_5_layer_call_fn_110711

inputs
unknown:	K
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_109832o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿK: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs
É
é
$__inference_signature_wrapper_110499
input_16
input_17!
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

unknown_11:	K

unknown_12:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinput_16input_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_109232o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_16:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_17
Û
Á
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_110923

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_109900

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿK:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs

¾
C__inference_model_3_layer_call_and_return_conditional_losses_109991

inputs
inputs_1(
model_5_109944:@
model_5_109946:@
model_5_109948:@
model_5_109950:@
model_5_109952:@
model_5_109954:@(
model_5_109956:@`
model_5_109958:`
model_5_109960:`
model_5_109962:`
model_5_109964:`
model_5_109966:`!
dense_5_109985:	K
dense_5_109987:
identity¢dense_5/StatefulPartitionedCall¢"dropout_17/StatefulPartitionedCall¢model_5/StatefulPartitionedCall¢!model_5/StatefulPartitionedCall_1 
model_5/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_5_109944model_5_109946model_5_109948model_5_109950model_5_109952model_5_109954model_5_109956model_5_109958model_5_109960model_5_109962model_5_109964model_5_109966*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109635Æ
!model_5/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_5_109944model_5_109946model_5_109948model_5_109950model_5_109952model_5_109954model_5_109956model_5_109958model_5_109960model_5_109962model_5_109964model_5_109966 ^model_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109635¦
tf.math.subtract_5/SubSub(model_5/StatefulPartitionedCall:output:0*model_5/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKg
tf.math.abs_5/AbsAbstf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKß
"dropout_17/StatefulPartitionedCallStatefulPartitionedCalltf.math.abs_5/Abs:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_109900
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_5_109985dense_5_109987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_109832w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp ^dense_5/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall ^model_5/StatefulPartitionedCall"^model_5/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall2F
!model_5/StatefulPartitionedCall_1!model_5/StatefulPartitionedCall_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
í
Ù
!__inference__wrapped_model_109232
input_16
input_17R
8model_3_model_5_conv2d_10_conv2d_readvariableop_resource:@G
9model_3_model_5_conv2d_10_biasadd_readvariableop_resource:@L
>model_3_model_5_batch_normalization_10_readvariableop_resource:@N
@model_3_model_5_batch_normalization_10_readvariableop_1_resource:@]
Omodel_3_model_5_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@_
Qmodel_3_model_5_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@R
8model_3_model_5_conv2d_11_conv2d_readvariableop_resource:@`G
9model_3_model_5_conv2d_11_biasadd_readvariableop_resource:`L
>model_3_model_5_batch_normalization_11_readvariableop_resource:`N
@model_3_model_5_batch_normalization_11_readvariableop_1_resource:`]
Omodel_3_model_5_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:`_
Qmodel_3_model_5_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:`A
.model_3_dense_5_matmul_readvariableop_resource:	K=
/model_3_dense_5_biasadd_readvariableop_resource:
identity¢&model_3/dense_5/BiasAdd/ReadVariableOp¢%model_3/dense_5/MatMul/ReadVariableOp¢Fmodel_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢Hmodel_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢Hmodel_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp¢Jmodel_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1¢5model_3/model_5/batch_normalization_10/ReadVariableOp¢7model_3/model_5/batch_normalization_10/ReadVariableOp_1¢7model_3/model_5/batch_normalization_10/ReadVariableOp_2¢7model_3/model_5/batch_normalization_10/ReadVariableOp_3¢Fmodel_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp¢Hmodel_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1¢Hmodel_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp¢Jmodel_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1¢5model_3/model_5/batch_normalization_11/ReadVariableOp¢7model_3/model_5/batch_normalization_11/ReadVariableOp_1¢7model_3/model_5/batch_normalization_11/ReadVariableOp_2¢7model_3/model_5/batch_normalization_11/ReadVariableOp_3¢0model_3/model_5/conv2d_10/BiasAdd/ReadVariableOp¢2model_3/model_5/conv2d_10/BiasAdd_1/ReadVariableOp¢/model_3/model_5/conv2d_10/Conv2D/ReadVariableOp¢1model_3/model_5/conv2d_10/Conv2D_1/ReadVariableOp¢0model_3/model_5/conv2d_11/BiasAdd/ReadVariableOp¢2model_3/model_5/conv2d_11/BiasAdd_1/ReadVariableOp¢/model_3/model_5/conv2d_11/Conv2D/ReadVariableOp¢1model_3/model_5/conv2d_11/Conv2D_1/ReadVariableOp°
/model_3/model_5/conv2d_10/Conv2D/ReadVariableOpReadVariableOp8model_3_model_5_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ð
 model_3/model_5/conv2d_10/Conv2DConv2Dinput_167model_3/model_5/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
paddingVALID*
strides
¦
0model_3/model_5/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp9model_3_model_5_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ë
!model_3/model_5/conv2d_10/BiasAddBiasAdd)model_3/model_5/conv2d_10/Conv2D:output:08model_3/model_5/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
model_3/model_5/conv2d_10/ReluRelu*model_3/model_5/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@°
5model_3/model_5/batch_normalization_10/ReadVariableOpReadVariableOp>model_3_model_5_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0´
7model_3/model_5/batch_normalization_10/ReadVariableOp_1ReadVariableOp@model_3_model_5_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ò
Fmodel_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_3_model_5_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
Hmodel_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_3_model_5_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
7model_3/model_5/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3,model_3/model_5/conv2d_10/Relu:activations:0=model_3/model_5/batch_normalization_10/ReadVariableOp:value:0?model_3/model_5/batch_normalization_10/ReadVariableOp_1:value:0Nmodel_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Pmodel_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿmm@:@:@:@:@:*
epsilon%o:*
is_training( ¦
#model_3/model_5/dropout_15/IdentityIdentity;model_3/model_5/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@Î
(model_3/model_5/max_pooling2d_10/MaxPoolMaxPool,model_3/model_5/dropout_15/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@*
ksize
*
paddingVALID*
strides
°
/model_3/model_5/conv2d_11/Conv2D/ReadVariableOpReadVariableOp8model_3_model_5_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0ù
 model_3/model_5/conv2d_11/Conv2DConv2D1model_3/model_5/max_pooling2d_10/MaxPool:output:07model_3/model_5/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
paddingVALID*
strides
¦
0model_3/model_5/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp9model_3_model_5_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ë
!model_3/model_5/conv2d_11/BiasAddBiasAdd)model_3/model_5/conv2d_11/Conv2D:output:08model_3/model_5/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
model_3/model_5/conv2d_11/ReluRelu*model_3/model_5/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `°
5model_3/model_5/batch_normalization_11/ReadVariableOpReadVariableOp>model_3_model_5_batch_normalization_11_readvariableop_resource*
_output_shapes
:`*
dtype0´
7model_3/model_5/batch_normalization_11/ReadVariableOp_1ReadVariableOp@model_3_model_5_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ò
Fmodel_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_3_model_5_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0Ö
Hmodel_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_3_model_5_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0
7model_3/model_5/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3,model_3/model_5/conv2d_11/Relu:activations:0=model_3/model_5/batch_normalization_11/ReadVariableOp:value:0?model_3/model_5/batch_normalization_11/ReadVariableOp_1:value:0Nmodel_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Pmodel_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  `:`:`:`:`:*
epsilon%o:*
is_training( ¦
#model_3/model_5/dropout_16/IdentityIdentity;model_3/model_5/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `Î
(model_3/model_5/max_pooling2d_11/MaxPoolMaxPool,model_3/model_5/dropout_16/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`*
ksize
*
paddingVALID*
strides
p
model_3/model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ%  ¼
!model_3/model_5/flatten_5/ReshapeReshape1model_3/model_5/max_pooling2d_11/MaxPool:output:0(model_3/model_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK²
1model_3/model_5/conv2d_10/Conv2D_1/ReadVariableOpReadVariableOp8model_3_model_5_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ô
"model_3/model_5/conv2d_10/Conv2D_1Conv2Dinput_179model_3/model_5/conv2d_10/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
paddingVALID*
strides
¨
2model_3/model_5/conv2d_10/BiasAdd_1/ReadVariableOpReadVariableOp9model_3_model_5_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ñ
#model_3/model_5/conv2d_10/BiasAdd_1BiasAdd+model_3/model_5/conv2d_10/Conv2D_1:output:0:model_3/model_5/conv2d_10/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
 model_3/model_5/conv2d_10/Relu_1Relu,model_3/model_5/conv2d_10/BiasAdd_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@²
7model_3/model_5/batch_normalization_10/ReadVariableOp_2ReadVariableOp>model_3_model_5_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0´
7model_3/model_5/batch_normalization_10/ReadVariableOp_3ReadVariableOp@model_3_model_5_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ô
Hmodel_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOpReadVariableOpOmodel_3_model_5_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ø
Jmodel_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpQmodel_3_model_5_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0©
9model_3/model_5/batch_normalization_10/FusedBatchNormV3_1FusedBatchNormV3.model_3/model_5/conv2d_10/Relu_1:activations:0?model_3/model_5/batch_normalization_10/ReadVariableOp_2:value:0?model_3/model_5/batch_normalization_10/ReadVariableOp_3:value:0Pmodel_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp:value:0Rmodel_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿmm@:@:@:@:@:*
epsilon%o:*
is_training( ª
%model_3/model_5/dropout_15/Identity_1Identity=model_3/model_5/batch_normalization_10/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@Ò
*model_3/model_5/max_pooling2d_10/MaxPool_1MaxPool.model_3/model_5/dropout_15/Identity_1:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@*
ksize
*
paddingVALID*
strides
²
1model_3/model_5/conv2d_11/Conv2D_1/ReadVariableOpReadVariableOp8model_3_model_5_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0ÿ
"model_3/model_5/conv2d_11/Conv2D_1Conv2D3model_3/model_5/max_pooling2d_10/MaxPool_1:output:09model_3/model_5/conv2d_11/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
paddingVALID*
strides
¨
2model_3/model_5/conv2d_11/BiasAdd_1/ReadVariableOpReadVariableOp9model_3_model_5_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ñ
#model_3/model_5/conv2d_11/BiasAdd_1BiasAdd+model_3/model_5/conv2d_11/Conv2D_1:output:0:model_3/model_5/conv2d_11/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
 model_3/model_5/conv2d_11/Relu_1Relu,model_3/model_5/conv2d_11/BiasAdd_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `²
7model_3/model_5/batch_normalization_11/ReadVariableOp_2ReadVariableOp>model_3_model_5_batch_normalization_11_readvariableop_resource*
_output_shapes
:`*
dtype0´
7model_3/model_5/batch_normalization_11/ReadVariableOp_3ReadVariableOp@model_3_model_5_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ô
Hmodel_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOpReadVariableOpOmodel_3_model_5_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0Ø
Jmodel_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpQmodel_3_model_5_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0©
9model_3/model_5/batch_normalization_11/FusedBatchNormV3_1FusedBatchNormV3.model_3/model_5/conv2d_11/Relu_1:activations:0?model_3/model_5/batch_normalization_11/ReadVariableOp_2:value:0?model_3/model_5/batch_normalization_11/ReadVariableOp_3:value:0Pmodel_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp:value:0Rmodel_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  `:`:`:`:`:*
epsilon%o:*
is_training( ª
%model_3/model_5/dropout_16/Identity_1Identity=model_3/model_5/batch_normalization_11/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `Ò
*model_3/model_5/max_pooling2d_11/MaxPool_1MaxPool.model_3/model_5/dropout_16/Identity_1:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`*
ksize
*
paddingVALID*
strides
r
!model_3/model_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ%  Â
#model_3/model_5/flatten_5/Reshape_1Reshape3model_3/model_5/max_pooling2d_11/MaxPool_1:output:0*model_3/model_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK²
model_3/tf.math.subtract_5/SubSub*model_3/model_5/flatten_5/Reshape:output:0,model_3/model_5/flatten_5/Reshape_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKw
model_3/tf.math.abs_5/AbsAbs"model_3/tf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKy
model_3/dropout_17/IdentityIdentitymodel_3/tf.math.abs_5/Abs:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
%model_3/dense_5/MatMul/ReadVariableOpReadVariableOp.model_3_dense_5_matmul_readvariableop_resource*
_output_shapes
:	K*
dtype0§
model_3/dense_5/MatMulMatMul$model_3/dropout_17/Identity:output:0-model_3/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model_3/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_3_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model_3/dense_5/BiasAddBiasAdd model_3/dense_5/MatMul:product:0.model_3/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
model_3/dense_5/SigmoidSigmoid model_3/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitymodel_3/dense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
NoOpNoOp'^model_3/dense_5/BiasAdd/ReadVariableOp&^model_3/dense_5/MatMul/ReadVariableOpG^model_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOpI^model_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1I^model_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOpK^model_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_16^model_3/model_5/batch_normalization_10/ReadVariableOp8^model_3/model_5/batch_normalization_10/ReadVariableOp_18^model_3/model_5/batch_normalization_10/ReadVariableOp_28^model_3/model_5/batch_normalization_10/ReadVariableOp_3G^model_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOpI^model_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1I^model_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOpK^model_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_16^model_3/model_5/batch_normalization_11/ReadVariableOp8^model_3/model_5/batch_normalization_11/ReadVariableOp_18^model_3/model_5/batch_normalization_11/ReadVariableOp_28^model_3/model_5/batch_normalization_11/ReadVariableOp_31^model_3/model_5/conv2d_10/BiasAdd/ReadVariableOp3^model_3/model_5/conv2d_10/BiasAdd_1/ReadVariableOp0^model_3/model_5/conv2d_10/Conv2D/ReadVariableOp2^model_3/model_5/conv2d_10/Conv2D_1/ReadVariableOp1^model_3/model_5/conv2d_11/BiasAdd/ReadVariableOp3^model_3/model_5/conv2d_11/BiasAdd_1/ReadVariableOp0^model_3/model_5/conv2d_11/Conv2D/ReadVariableOp2^model_3/model_5/conv2d_11/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2P
&model_3/dense_5/BiasAdd/ReadVariableOp&model_3/dense_5/BiasAdd/ReadVariableOp2N
%model_3/dense_5/MatMul/ReadVariableOp%model_3/dense_5/MatMul/ReadVariableOp2
Fmodel_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOpFmodel_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2
Hmodel_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Hmodel_3/model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12
Hmodel_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOpHmodel_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp2
Jmodel_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1Jmodel_3/model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_12n
5model_3/model_5/batch_normalization_10/ReadVariableOp5model_3/model_5/batch_normalization_10/ReadVariableOp2r
7model_3/model_5/batch_normalization_10/ReadVariableOp_17model_3/model_5/batch_normalization_10/ReadVariableOp_12r
7model_3/model_5/batch_normalization_10/ReadVariableOp_27model_3/model_5/batch_normalization_10/ReadVariableOp_22r
7model_3/model_5/batch_normalization_10/ReadVariableOp_37model_3/model_5/batch_normalization_10/ReadVariableOp_32
Fmodel_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOpFmodel_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2
Hmodel_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Hmodel_3/model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12
Hmodel_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOpHmodel_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp2
Jmodel_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1Jmodel_3/model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_12n
5model_3/model_5/batch_normalization_11/ReadVariableOp5model_3/model_5/batch_normalization_11/ReadVariableOp2r
7model_3/model_5/batch_normalization_11/ReadVariableOp_17model_3/model_5/batch_normalization_11/ReadVariableOp_12r
7model_3/model_5/batch_normalization_11/ReadVariableOp_27model_3/model_5/batch_normalization_11/ReadVariableOp_22r
7model_3/model_5/batch_normalization_11/ReadVariableOp_37model_3/model_5/batch_normalization_11/ReadVariableOp_32d
0model_3/model_5/conv2d_10/BiasAdd/ReadVariableOp0model_3/model_5/conv2d_10/BiasAdd/ReadVariableOp2h
2model_3/model_5/conv2d_10/BiasAdd_1/ReadVariableOp2model_3/model_5/conv2d_10/BiasAdd_1/ReadVariableOp2b
/model_3/model_5/conv2d_10/Conv2D/ReadVariableOp/model_3/model_5/conv2d_10/Conv2D/ReadVariableOp2f
1model_3/model_5/conv2d_10/Conv2D_1/ReadVariableOp1model_3/model_5/conv2d_10/Conv2D_1/ReadVariableOp2d
0model_3/model_5/conv2d_11/BiasAdd/ReadVariableOp0model_3/model_5/conv2d_11/BiasAdd/ReadVariableOp2h
2model_3/model_5/conv2d_11/BiasAdd_1/ReadVariableOp2model_3/model_5/conv2d_11/BiasAdd_1/ReadVariableOp2b
/model_3/model_5/conv2d_11/Conv2D/ReadVariableOp/model_3/model_5/conv2d_11/Conv2D/ReadVariableOp2f
1model_3/model_5/conv2d_11/Conv2D_1/ReadVariableOp1model_3/model_5/conv2d_11/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_16:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_17


õ
C__inference_dense_5_layer_call_and_return_conditional_losses_109832

inputs1
matmul_readvariableop_resource:	K-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	K*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs
åÞ
Ç
C__inference_model_3_layer_call_and_return_conditional_losses_110463
inputs_0
inputs_1J
0model_5_conv2d_10_conv2d_readvariableop_resource:@?
1model_5_conv2d_10_biasadd_readvariableop_resource:@D
6model_5_batch_normalization_10_readvariableop_resource:@F
8model_5_batch_normalization_10_readvariableop_1_resource:@U
Gmodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@W
Imodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@J
0model_5_conv2d_11_conv2d_readvariableop_resource:@`?
1model_5_conv2d_11_biasadd_readvariableop_resource:`D
6model_5_batch_normalization_11_readvariableop_resource:`F
8model_5_batch_normalization_11_readvariableop_1_resource:`U
Gmodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:`W
Imodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:`9
&dense_5_matmul_readvariableop_resource:	K5
'dense_5_biasadd_readvariableop_resource:
identity¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢-model_5/batch_normalization_10/AssignNewValue¢/model_5/batch_normalization_10/AssignNewValue_1¢/model_5/batch_normalization_10/AssignNewValue_2¢/model_5/batch_normalization_10/AssignNewValue_3¢>model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢@model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢@model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp¢Bmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1¢-model_5/batch_normalization_10/ReadVariableOp¢/model_5/batch_normalization_10/ReadVariableOp_1¢/model_5/batch_normalization_10/ReadVariableOp_2¢/model_5/batch_normalization_10/ReadVariableOp_3¢-model_5/batch_normalization_11/AssignNewValue¢/model_5/batch_normalization_11/AssignNewValue_1¢/model_5/batch_normalization_11/AssignNewValue_2¢/model_5/batch_normalization_11/AssignNewValue_3¢>model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp¢@model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1¢@model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp¢Bmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1¢-model_5/batch_normalization_11/ReadVariableOp¢/model_5/batch_normalization_11/ReadVariableOp_1¢/model_5/batch_normalization_11/ReadVariableOp_2¢/model_5/batch_normalization_11/ReadVariableOp_3¢(model_5/conv2d_10/BiasAdd/ReadVariableOp¢*model_5/conv2d_10/BiasAdd_1/ReadVariableOp¢'model_5/conv2d_10/Conv2D/ReadVariableOp¢)model_5/conv2d_10/Conv2D_1/ReadVariableOp¢(model_5/conv2d_11/BiasAdd/ReadVariableOp¢*model_5/conv2d_11/BiasAdd_1/ReadVariableOp¢'model_5/conv2d_11/Conv2D/ReadVariableOp¢)model_5/conv2d_11/Conv2D_1/ReadVariableOp 
'model_5/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_5_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0À
model_5/conv2d_10/Conv2DConv2Dinputs_0/model_5/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
paddingVALID*
strides

(model_5/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
model_5/conv2d_10/BiasAddBiasAdd!model_5/conv2d_10/Conv2D:output:00model_5/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@|
model_5/conv2d_10/ReluRelu"model_5/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@ 
-model_5/batch_normalization_10/ReadVariableOpReadVariableOp6model_5_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0¤
/model_5/batch_normalization_10/ReadVariableOp_1ReadVariableOp8model_5_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0Â
>model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Æ
@model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ý
/model_5/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3$model_5/conv2d_10/Relu:activations:05model_5/batch_normalization_10/ReadVariableOp:value:07model_5/batch_normalization_10/ReadVariableOp_1:value:0Fmodel_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿmm@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<¬
-model_5/batch_normalization_10/AssignNewValueAssignVariableOpGmodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_resource<model_5/batch_normalization_10/FusedBatchNormV3:batch_mean:0?^model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0¶
/model_5/batch_normalization_10/AssignNewValue_1AssignVariableOpImodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource@model_5/batch_normalization_10/FusedBatchNormV3:batch_variance:0A^model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0e
 model_5/dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¿
model_5/dropout_15/dropout/MulMul3model_5/batch_normalization_10/FusedBatchNormV3:y:0)model_5/dropout_15/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
 model_5/dropout_15/dropout/ShapeShape3model_5/batch_normalization_10/FusedBatchNormV3:y:0*
T0*
_output_shapes
:º
7model_5/dropout_15/dropout/random_uniform/RandomUniformRandomUniform)model_5/dropout_15/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
dtype0n
)model_5/dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ç
'model_5/dropout_15/dropout/GreaterEqualGreaterEqual@model_5/dropout_15/dropout/random_uniform/RandomUniform:output:02model_5/dropout_15/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
model_5/dropout_15/dropout/CastCast+model_5/dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@ª
 model_5/dropout_15/dropout/Mul_1Mul"model_5/dropout_15/dropout/Mul:z:0#model_5/dropout_15/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@¾
 model_5/max_pooling2d_10/MaxPoolMaxPool$model_5/dropout_15/dropout/Mul_1:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@*
ksize
*
paddingVALID*
strides
 
'model_5/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_5_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0á
model_5/conv2d_11/Conv2DConv2D)model_5/max_pooling2d_10/MaxPool:output:0/model_5/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
paddingVALID*
strides

(model_5/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0³
model_5/conv2d_11/BiasAddBiasAdd!model_5/conv2d_11/Conv2D:output:00model_5/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `|
model_5/conv2d_11/ReluRelu"model_5/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ` 
-model_5/batch_normalization_11/ReadVariableOpReadVariableOp6model_5_batch_normalization_11_readvariableop_resource*
_output_shapes
:`*
dtype0¤
/model_5/batch_normalization_11/ReadVariableOp_1ReadVariableOp8model_5_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:`*
dtype0Â
>model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0Æ
@model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ý
/model_5/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3$model_5/conv2d_11/Relu:activations:05model_5/batch_normalization_11/ReadVariableOp:value:07model_5/batch_normalization_11/ReadVariableOp_1:value:0Fmodel_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  `:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
×#<¬
-model_5/batch_normalization_11/AssignNewValueAssignVariableOpGmodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_resource<model_5/batch_normalization_11/FusedBatchNormV3:batch_mean:0?^model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0¶
/model_5/batch_normalization_11/AssignNewValue_1AssignVariableOpImodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource@model_5/batch_normalization_11/FusedBatchNormV3:batch_variance:0A^model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0e
 model_5/dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¿
model_5/dropout_16/dropout/MulMul3model_5/batch_normalization_11/FusedBatchNormV3:y:0)model_5/dropout_16/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
 model_5/dropout_16/dropout/ShapeShape3model_5/batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:º
7model_5/dropout_16/dropout/random_uniform/RandomUniformRandomUniform)model_5/dropout_16/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
dtype0n
)model_5/dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ç
'model_5/dropout_16/dropout/GreaterEqualGreaterEqual@model_5/dropout_16/dropout/random_uniform/RandomUniform:output:02model_5/dropout_16/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
model_5/dropout_16/dropout/CastCast+model_5/dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `ª
 model_5/dropout_16/dropout/Mul_1Mul"model_5/dropout_16/dropout/Mul:z:0#model_5/dropout_16/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `¾
 model_5/max_pooling2d_11/MaxPoolMaxPool$model_5/dropout_16/dropout/Mul_1:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`*
ksize
*
paddingVALID*
strides
h
model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ%  ¤
model_5/flatten_5/ReshapeReshape)model_5/max_pooling2d_11/MaxPool:output:0 model_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK¢
)model_5/conv2d_10/Conv2D_1/ReadVariableOpReadVariableOp0model_5_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
model_5/conv2d_10/Conv2D_1Conv2Dinputs_11model_5/conv2d_10/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
paddingVALID*
strides

*model_5/conv2d_10/BiasAdd_1/ReadVariableOpReadVariableOp1model_5_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¹
model_5/conv2d_10/BiasAdd_1BiasAdd#model_5/conv2d_10/Conv2D_1:output:02model_5/conv2d_10/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
model_5/conv2d_10/Relu_1Relu$model_5/conv2d_10/BiasAdd_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@¢
/model_5/batch_normalization_10/ReadVariableOp_2ReadVariableOp6model_5_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0¤
/model_5/batch_normalization_10/ReadVariableOp_3ReadVariableOp8model_5_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0ô
@model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGmodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_resource.^model_5/batch_normalization_10/AssignNewValue*
_output_shapes
:@*
dtype0ú
Bmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpImodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource0^model_5/batch_normalization_10/AssignNewValue_1*
_output_shapes
:@*
dtype0
1model_5/batch_normalization_10/FusedBatchNormV3_1FusedBatchNormV3&model_5/conv2d_10/Relu_1:activations:07model_5/batch_normalization_10/ReadVariableOp_2:value:07model_5/batch_normalization_10/ReadVariableOp_3:value:0Hmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp:value:0Jmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿmm@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<â
/model_5/batch_normalization_10/AssignNewValue_2AssignVariableOpGmodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_resource>model_5/batch_normalization_10/FusedBatchNormV3_1:batch_mean:0.^model_5/batch_normalization_10/AssignNewValueA^model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0ì
/model_5/batch_normalization_10/AssignNewValue_3AssignVariableOpImodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resourceBmodel_5/batch_normalization_10/FusedBatchNormV3_1:batch_variance:00^model_5/batch_normalization_10/AssignNewValue_1C^model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0g
"model_5/dropout_15/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Å
 model_5/dropout_15/dropout_1/MulMul5model_5/batch_normalization_10/FusedBatchNormV3_1:y:0+model_5/dropout_15/dropout_1/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
"model_5/dropout_15/dropout_1/ShapeShape5model_5/batch_normalization_10/FusedBatchNormV3_1:y:0*
T0*
_output_shapes
:¾
9model_5/dropout_15/dropout_1/random_uniform/RandomUniformRandomUniform+model_5/dropout_15/dropout_1/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
dtype0p
+model_5/dropout_15/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>í
)model_5/dropout_15/dropout_1/GreaterEqualGreaterEqualBmodel_5/dropout_15/dropout_1/random_uniform/RandomUniform:output:04model_5/dropout_15/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@¡
!model_5/dropout_15/dropout_1/CastCast-model_5/dropout_15/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@°
"model_5/dropout_15/dropout_1/Mul_1Mul$model_5/dropout_15/dropout_1/Mul:z:0%model_5/dropout_15/dropout_1/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@Â
"model_5/max_pooling2d_10/MaxPool_1MaxPool&model_5/dropout_15/dropout_1/Mul_1:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@*
ksize
*
paddingVALID*
strides
¢
)model_5/conv2d_11/Conv2D_1/ReadVariableOpReadVariableOp0model_5_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0ç
model_5/conv2d_11/Conv2D_1Conv2D+model_5/max_pooling2d_10/MaxPool_1:output:01model_5/conv2d_11/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
paddingVALID*
strides

*model_5/conv2d_11/BiasAdd_1/ReadVariableOpReadVariableOp1model_5_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0¹
model_5/conv2d_11/BiasAdd_1BiasAdd#model_5/conv2d_11/Conv2D_1:output:02model_5/conv2d_11/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
model_5/conv2d_11/Relu_1Relu$model_5/conv2d_11/BiasAdd_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `¢
/model_5/batch_normalization_11/ReadVariableOp_2ReadVariableOp6model_5_batch_normalization_11_readvariableop_resource*
_output_shapes
:`*
dtype0¤
/model_5/batch_normalization_11/ReadVariableOp_3ReadVariableOp8model_5_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:`*
dtype0ô
@model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGmodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_resource.^model_5/batch_normalization_11/AssignNewValue*
_output_shapes
:`*
dtype0ú
Bmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpImodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource0^model_5/batch_normalization_11/AssignNewValue_1*
_output_shapes
:`*
dtype0
1model_5/batch_normalization_11/FusedBatchNormV3_1FusedBatchNormV3&model_5/conv2d_11/Relu_1:activations:07model_5/batch_normalization_11/ReadVariableOp_2:value:07model_5/batch_normalization_11/ReadVariableOp_3:value:0Hmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp:value:0Jmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  `:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
×#<â
/model_5/batch_normalization_11/AssignNewValue_2AssignVariableOpGmodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_resource>model_5/batch_normalization_11/FusedBatchNormV3_1:batch_mean:0.^model_5/batch_normalization_11/AssignNewValueA^model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0ì
/model_5/batch_normalization_11/AssignNewValue_3AssignVariableOpImodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceBmodel_5/batch_normalization_11/FusedBatchNormV3_1:batch_variance:00^model_5/batch_normalization_11/AssignNewValue_1C^model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0g
"model_5/dropout_16/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Å
 model_5/dropout_16/dropout_1/MulMul5model_5/batch_normalization_11/FusedBatchNormV3_1:y:0+model_5/dropout_16/dropout_1/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
"model_5/dropout_16/dropout_1/ShapeShape5model_5/batch_normalization_11/FusedBatchNormV3_1:y:0*
T0*
_output_shapes
:¾
9model_5/dropout_16/dropout_1/random_uniform/RandomUniformRandomUniform+model_5/dropout_16/dropout_1/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
dtype0p
+model_5/dropout_16/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>í
)model_5/dropout_16/dropout_1/GreaterEqualGreaterEqualBmodel_5/dropout_16/dropout_1/random_uniform/RandomUniform:output:04model_5/dropout_16/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `¡
!model_5/dropout_16/dropout_1/CastCast-model_5/dropout_16/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `°
"model_5/dropout_16/dropout_1/Mul_1Mul$model_5/dropout_16/dropout_1/Mul:z:0%model_5/dropout_16/dropout_1/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `Â
"model_5/max_pooling2d_11/MaxPool_1MaxPool&model_5/dropout_16/dropout_1/Mul_1:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`*
ksize
*
paddingVALID*
strides
j
model_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ%  ª
model_5/flatten_5/Reshape_1Reshape+model_5/max_pooling2d_11/MaxPool_1:output:0"model_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
tf.math.subtract_5/SubSub"model_5/flatten_5/Reshape:output:0$model_5/flatten_5/Reshape_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKg
tf.math.abs_5/AbsAbstf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK]
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_17/dropout/MulMultf.math.abs_5/Abs:y:0!dropout_17/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK]
dropout_17/dropout/ShapeShapetf.math.abs_5/Abs:y:0*
T0*
_output_shapes
:£
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
dtype0f
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>È
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	K*
dtype0
dense_5/MatMulMatMuldropout_17/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp.^model_5/batch_normalization_10/AssignNewValue0^model_5/batch_normalization_10/AssignNewValue_10^model_5/batch_normalization_10/AssignNewValue_20^model_5/batch_normalization_10/AssignNewValue_3?^model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOpA^model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1A^model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOpC^model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1.^model_5/batch_normalization_10/ReadVariableOp0^model_5/batch_normalization_10/ReadVariableOp_10^model_5/batch_normalization_10/ReadVariableOp_20^model_5/batch_normalization_10/ReadVariableOp_3.^model_5/batch_normalization_11/AssignNewValue0^model_5/batch_normalization_11/AssignNewValue_10^model_5/batch_normalization_11/AssignNewValue_20^model_5/batch_normalization_11/AssignNewValue_3?^model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOpA^model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1A^model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOpC^model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1.^model_5/batch_normalization_11/ReadVariableOp0^model_5/batch_normalization_11/ReadVariableOp_10^model_5/batch_normalization_11/ReadVariableOp_20^model_5/batch_normalization_11/ReadVariableOp_3)^model_5/conv2d_10/BiasAdd/ReadVariableOp+^model_5/conv2d_10/BiasAdd_1/ReadVariableOp(^model_5/conv2d_10/Conv2D/ReadVariableOp*^model_5/conv2d_10/Conv2D_1/ReadVariableOp)^model_5/conv2d_11/BiasAdd/ReadVariableOp+^model_5/conv2d_11/BiasAdd_1/ReadVariableOp(^model_5/conv2d_11/Conv2D/ReadVariableOp*^model_5/conv2d_11/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2^
-model_5/batch_normalization_10/AssignNewValue-model_5/batch_normalization_10/AssignNewValue2b
/model_5/batch_normalization_10/AssignNewValue_1/model_5/batch_normalization_10/AssignNewValue_12b
/model_5/batch_normalization_10/AssignNewValue_2/model_5/batch_normalization_10/AssignNewValue_22b
/model_5/batch_normalization_10/AssignNewValue_3/model_5/batch_normalization_10/AssignNewValue_32
>model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp>model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2
@model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1@model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12
@model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp@model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp2
Bmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1Bmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_12^
-model_5/batch_normalization_10/ReadVariableOp-model_5/batch_normalization_10/ReadVariableOp2b
/model_5/batch_normalization_10/ReadVariableOp_1/model_5/batch_normalization_10/ReadVariableOp_12b
/model_5/batch_normalization_10/ReadVariableOp_2/model_5/batch_normalization_10/ReadVariableOp_22b
/model_5/batch_normalization_10/ReadVariableOp_3/model_5/batch_normalization_10/ReadVariableOp_32^
-model_5/batch_normalization_11/AssignNewValue-model_5/batch_normalization_11/AssignNewValue2b
/model_5/batch_normalization_11/AssignNewValue_1/model_5/batch_normalization_11/AssignNewValue_12b
/model_5/batch_normalization_11/AssignNewValue_2/model_5/batch_normalization_11/AssignNewValue_22b
/model_5/batch_normalization_11/AssignNewValue_3/model_5/batch_normalization_11/AssignNewValue_32
>model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp>model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2
@model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1@model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12
@model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp@model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp2
Bmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1Bmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_12^
-model_5/batch_normalization_11/ReadVariableOp-model_5/batch_normalization_11/ReadVariableOp2b
/model_5/batch_normalization_11/ReadVariableOp_1/model_5/batch_normalization_11/ReadVariableOp_12b
/model_5/batch_normalization_11/ReadVariableOp_2/model_5/batch_normalization_11/ReadVariableOp_22b
/model_5/batch_normalization_11/ReadVariableOp_3/model_5/batch_normalization_11/ReadVariableOp_32T
(model_5/conv2d_10/BiasAdd/ReadVariableOp(model_5/conv2d_10/BiasAdd/ReadVariableOp2X
*model_5/conv2d_10/BiasAdd_1/ReadVariableOp*model_5/conv2d_10/BiasAdd_1/ReadVariableOp2R
'model_5/conv2d_10/Conv2D/ReadVariableOp'model_5/conv2d_10/Conv2D/ReadVariableOp2V
)model_5/conv2d_10/Conv2D_1/ReadVariableOp)model_5/conv2d_10/Conv2D_1/ReadVariableOp2T
(model_5/conv2d_11/BiasAdd/ReadVariableOp(model_5/conv2d_11/BiasAdd/ReadVariableOp2X
*model_5/conv2d_11/BiasAdd_1/ReadVariableOp*model_5/conv2d_11/BiasAdd_1/ReadVariableOp2R
'model_5/conv2d_11/Conv2D/ReadVariableOp'model_5/conv2d_11/Conv2D/ReadVariableOp2V
)model_5/conv2d_11/Conv2D_1/ReadVariableOp)model_5/conv2d_11/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
inputs/1
Û
Á
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_110804

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

þ
E__inference_conv2d_10_layer_call_and_return_conditional_losses_110742

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
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
:ÿÿÿÿÿÿÿÿÿmm@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
½
M
1__inference_max_pooling2d_10_layer_call_fn_110836

inputs
identityÝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_109305
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í

C__inference_model_3_layer_call_and_return_conditional_losses_110107
input_16
input_17(
model_5_110060:@
model_5_110062:@
model_5_110064:@
model_5_110066:@
model_5_110068:@
model_5_110070:@(
model_5_110072:@`
model_5_110074:`
model_5_110076:`
model_5_110078:`
model_5_110080:`
model_5_110082:`!
dense_5_110101:	K
dense_5_110103:
identity¢dense_5/StatefulPartitionedCall¢model_5/StatefulPartitionedCall¢!model_5/StatefulPartitionedCall_1¦
model_5/StatefulPartitionedCallStatefulPartitionedCallinput_16model_5_110060model_5_110062model_5_110064model_5_110066model_5_110068model_5_110070model_5_110072model_5_110074model_5_110076model_5_110078model_5_110080model_5_110082*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109468¨
!model_5/StatefulPartitionedCall_1StatefulPartitionedCallinput_17model_5_110060model_5_110062model_5_110064model_5_110066model_5_110068model_5_110070model_5_110072model_5_110074model_5_110076model_5_110078model_5_110080model_5_110082*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109468¦
tf.math.subtract_5/SubSub(model_5/StatefulPartitionedCall:output:0*model_5/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKg
tf.math.abs_5/AbsAbstf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÏ
dropout_17/PartitionedCallPartitionedCalltf.math.abs_5/Abs:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_109819
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_5_110101dense_5_110103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_109832w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp ^dense_5/StatefulPartitionedCall ^model_5/StatefulPartitionedCall"^model_5/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall2F
!model_5/StatefulPartitionedCall_1!model_5/StatefulPartitionedCall_1:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_16:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_17

h
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_110841

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

¤
(__inference_model_5_layer_call_fn_110557

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
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109635p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
Û
Á
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_109361

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
	
Ò
7__inference_batch_normalization_11_layer_call_fn_110887

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_109361
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
´

e
F__inference_dropout_16_layer_call_and_return_conditional_losses_109521

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  `:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
 
_user_specified_nameinputs
´

e
F__inference_dropout_16_layer_call_and_return_conditional_losses_110950

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  `:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
 
_user_specified_nameinputs
ï

*__inference_conv2d_11_layer_call_fn_110850

inputs!
unknown:@`
	unknown_0:`
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_109436w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ$$@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_109381

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_110690

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿK:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs
Í

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_110786

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_109305

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¦
(__inference_model_5_layer_call_fn_109691
input_18!
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
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109635p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_18
Ç
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_109465

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ%  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

`:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`
 
_user_specified_nameinputs
Ö-
Ø
C__inference_model_5_layer_call_and_return_conditional_losses_109765
input_18*
conv2d_10_109731:@
conv2d_10_109733:@+
batch_normalization_10_109736:@+
batch_normalization_10_109738:@+
batch_normalization_10_109740:@+
batch_normalization_10_109742:@*
conv2d_11_109747:@`
conv2d_11_109749:`+
batch_normalization_11_109752:`+
batch_normalization_11_109754:`+
batch_normalization_11_109756:`+
batch_normalization_11_109758:`
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢"dropout_15/StatefulPartitionedCall¢"dropout_16/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_18conv2d_10_109731conv2d_10_109733*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_109402
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_109736batch_normalization_10_109738batch_normalization_10_109740batch_normalization_10_109742*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_109285
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_109554ø
 max_pooling2d_10/PartitionedCallPartitionedCall+dropout_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_109305¢
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_109747conv2d_11_109749*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_109436
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_109752batch_normalization_11_109754batch_normalization_11_109756batch_normalization_11_109758*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_109361­
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0#^dropout_15/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_109521ø
 max_pooling2d_11/PartitionedCallPartitionedCall+dropout_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_109381á
flatten_5/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_109465r
IdentityIdentity"flatten_5/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKº
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_18
	
Ò
7__inference_batch_normalization_10_layer_call_fn_110768

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_109285
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_110905

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
¨
G
+__inference_dropout_17_layer_call_fn_110680

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_109819a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿK:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs
Å

C__inference_model_3_layer_call_and_return_conditional_losses_109839

inputs
inputs_1(
model_5_109774:@
model_5_109776:@
model_5_109778:@
model_5_109780:@
model_5_109782:@
model_5_109784:@(
model_5_109786:@`
model_5_109788:`
model_5_109790:`
model_5_109792:`
model_5_109794:`
model_5_109796:`!
dense_5_109833:	K
dense_5_109835:
identity¢dense_5/StatefulPartitionedCall¢model_5/StatefulPartitionedCall¢!model_5/StatefulPartitionedCall_1¤
model_5/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_5_109774model_5_109776model_5_109778model_5_109780model_5_109782model_5_109784model_5_109786model_5_109788model_5_109790model_5_109792model_5_109794model_5_109796*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109468¨
!model_5/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_5_109774model_5_109776model_5_109778model_5_109780model_5_109782model_5_109784model_5_109786model_5_109788model_5_109790model_5_109792model_5_109794model_5_109796*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109468¦
tf.math.subtract_5/SubSub(model_5/StatefulPartitionedCall:output:0*model_5/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKg
tf.math.abs_5/AbsAbstf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÏ
dropout_17/PartitionedCallPartitionedCalltf.math.abs_5/Abs:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_109819
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_5_109833dense_5_109835*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_109832w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp ^dense_5/StatefulPartitionedCall ^model_5/StatefulPartitionedCall"^model_5/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall2F
!model_5/StatefulPartitionedCall_1!model_5/StatefulPartitionedCall_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ï
í
(__inference_model_3_layer_call_fn_110196
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

unknown_11:	K

unknown_12:
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_109839o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
inputs/1
ú\
Ò
C__inference_model_5_layer_call_and_return_conditional_losses_110675

inputsB
(conv2d_10_conv2d_readvariableop_resource:@7
)conv2d_10_biasadd_readvariableop_resource:@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_11_conv2d_readvariableop_resource:@`7
)conv2d_11_biasadd_readvariableop_resource:`<
.batch_normalization_11_readvariableop_resource:`>
0batch_normalization_11_readvariableop_1_resource:`M
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:`O
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:`
identity¢%batch_normalization_10/AssignNewValue¢'batch_normalization_10/AssignNewValue_1¢6batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_10/ReadVariableOp¢'batch_normalization_10/ReadVariableOp_1¢%batch_normalization_11/AssignNewValue¢'batch_normalization_11/AssignNewValue_1¢6batch_normalization_11/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_11/ReadVariableOp¢'batch_normalization_11/ReadVariableOp_1¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp¢ conv2d_11/BiasAdd/ReadVariableOp¢conv2d_11/Conv2D/ReadVariableOp
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0®
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
paddingVALID*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@l
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Í
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿmm@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0]
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout_15/dropout/MulMul+batch_normalization_10/FusedBatchNormV3:y:0!dropout_15/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@s
dropout_15/dropout/ShapeShape+batch_normalization_10/FusedBatchNormV3:y:0*
T0*
_output_shapes
:ª
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
dtype0f
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ï
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@®
max_pooling2d_10/MaxPoolMaxPooldropout_15/dropout/Mul_1:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@*
ksize
*
paddingVALID*
strides

conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0É
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
paddingVALID*
strides

 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `l
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:`*
dtype0
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:`*
dtype0²
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0¶
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0Í
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  `:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
×#<
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0]
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout_16/dropout/MulMul+batch_normalization_11/FusedBatchNormV3:y:0!dropout_16/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `s
dropout_16/dropout/ShapeShape+batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:ª
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
dtype0f
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ï
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `®
max_pooling2d_11/MaxPoolMaxPooldropout_16/dropout/Mul_1:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`*
ksize
*
paddingVALID*
strides
`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ%  
flatten_5/ReshapeReshape!max_pooling2d_11/MaxPool:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKj
IdentityIdentityflatten_5/Reshape:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ï
í
(__inference_model_3_layer_call_fn_109870
input_16
input_17!
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

unknown_11:	K

unknown_12:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_16input_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_109839o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_16:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_17
ó

*__inference_conv2d_10_layer_call_fn_110731

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_109402w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ý[
Ã
__inference__traced_save_111133
file_prefix-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop3
/savev2_weights_intermediate_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop6
2savev2_adam_conv2d_10_kernel_m_read_readvariableop4
0savev2_adam_conv2d_10_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_10_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_10_beta_m_read_readvariableop6
2savev2_adam_conv2d_11_kernel_m_read_readvariableop4
0savev2_adam_conv2d_11_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_m_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop6
2savev2_adam_conv2d_10_kernel_v_read_readvariableop4
0savev2_adam_conv2d_10_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_10_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_10_beta_v_read_readvariableop6
2savev2_adam_conv2d_11_kernel_v_read_readvariableop4
0savev2_adam_conv2d_11_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_v_read_readvariableop
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
: ó
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*
valueB/B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/2/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHË
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ý
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop/savev2_weights_intermediate_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop2savev2_adam_conv2d_10_kernel_m_read_readvariableop0savev2_adam_conv2d_10_bias_m_read_readvariableop>savev2_adam_batch_normalization_10_gamma_m_read_readvariableop=savev2_adam_batch_normalization_10_beta_m_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableop>savev2_adam_batch_normalization_11_gamma_m_read_readvariableop=savev2_adam_batch_normalization_11_beta_m_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop2savev2_adam_conv2d_10_kernel_v_read_readvariableop0savev2_adam_conv2d_10_bias_v_read_readvariableop>savev2_adam_batch_normalization_10_gamma_v_read_readvariableop=savev2_adam_batch_normalization_10_beta_v_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableop>savev2_adam_batch_normalization_11_gamma_v_read_readvariableop=savev2_adam_batch_normalization_11_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/	
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

identity_1Identity_1:output:0*Ô
_input_shapesÂ
¿: :	K:: : : : :@:@:@:@:@:@:@`:`:`:`:`:`: : : : : : : : :	K::@:@:@:@:@`:`:`:`:	K::@:@:@:@:@`:`:`:`: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	K: 
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
:	K: 
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
:	K: &
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
ë
í
(__inference_model_3_layer_call_fn_110230
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

unknown_11:	K

unknown_12:
identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_109991o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
inputs/1
ù
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_110938

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  `:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
 
_user_specified_nameinputs
Ð-
Ö
C__inference_model_5_layer_call_and_return_conditional_losses_109635

inputs*
conv2d_10_109601:@
conv2d_10_109603:@+
batch_normalization_10_109606:@+
batch_normalization_10_109608:@+
batch_normalization_10_109610:@+
batch_normalization_10_109612:@*
conv2d_11_109617:@`
conv2d_11_109619:`+
batch_normalization_11_109622:`+
batch_normalization_11_109624:`+
batch_normalization_11_109626:`+
batch_normalization_11_109628:`
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢"dropout_15/StatefulPartitionedCall¢"dropout_16/StatefulPartitionedCallÿ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_109601conv2d_10_109603*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_109402
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_109606batch_normalization_10_109608batch_normalization_10_109610batch_normalization_10_109612*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_109285
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_109554ø
 max_pooling2d_10/PartitionedCallPartitionedCall+dropout_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_109305¢
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_109617conv2d_11_109619*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_109436
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_109622batch_normalization_11_109624batch_normalization_11_109626batch_normalization_11_109628*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_109361­
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0#^dropout_15/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_109521ø
 max_pooling2d_11/PartitionedCallPartitionedCall+dropout_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_109381á
flatten_5/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_109465r
IdentityIdentity"flatten_5/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKº
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
Ä
G
+__inference_dropout_15_layer_call_fn_110809

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_109422h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿmm@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_110960

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
d
+__inference_dropout_17_layer_call_fn_110685

inputs
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_109900p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿK22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs
Í

R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_109330

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`:`:`:`:`:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs

À
C__inference_model_3_layer_call_and_return_conditional_losses_110158
input_16
input_17(
model_5_110111:@
model_5_110113:@
model_5_110115:@
model_5_110117:@
model_5_110119:@
model_5_110121:@(
model_5_110123:@`
model_5_110125:`
model_5_110127:`
model_5_110129:`
model_5_110131:`
model_5_110133:`!
dense_5_110152:	K
dense_5_110154:
identity¢dense_5/StatefulPartitionedCall¢"dropout_17/StatefulPartitionedCall¢model_5/StatefulPartitionedCall¢!model_5/StatefulPartitionedCall_1¢
model_5/StatefulPartitionedCallStatefulPartitionedCallinput_16model_5_110111model_5_110113model_5_110115model_5_110117model_5_110119model_5_110121model_5_110123model_5_110125model_5_110127model_5_110129model_5_110131model_5_110133*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109635Æ
!model_5/StatefulPartitionedCall_1StatefulPartitionedCallinput_17model_5_110111model_5_110113model_5_110115model_5_110117model_5_110119model_5_110121model_5_110123model_5_110125model_5_110127model_5_110129model_5_110131model_5_110133 ^model_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109635¦
tf.math.subtract_5/SubSub(model_5/StatefulPartitionedCall:output:0*model_5/StatefulPartitionedCall_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKg
tf.math.abs_5/AbsAbstf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKß
"dropout_17/StatefulPartitionedCallStatefulPartitionedCalltf.math.abs_5/Abs:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_109900
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_5_110152dense_5_110154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_109832w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp ^dense_5/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall ^model_5/StatefulPartitionedCall"^model_5/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall2F
!model_5/StatefulPartitionedCall_1!model_5/StatefulPartitionedCall_1:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_16:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_17
	
Ò
7__inference_batch_normalization_10_layer_call_fn_110755

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_109254
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

þ
E__inference_conv2d_10_layer_call_and_return_conditional_losses_109402

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
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
:ÿÿÿÿÿÿÿÿÿmm@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
´

e
F__inference_dropout_15_layer_call_and_return_conditional_losses_110831

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿmm@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
 
_user_specified_nameinputs
Û
Á
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_109285

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ä
G
+__inference_dropout_16_layer_call_fn_110928

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_109456h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  `:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
 
_user_specified_nameinputs
´
F
*__inference_flatten_5_layer_call_fn_110965

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_109465a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

`:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`
 
_user_specified_nameinputs


õ
C__inference_dense_5_layer_call_and_return_conditional_losses_110722

inputs1
matmul_readvariableop_resource:	K-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	K*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs
´

e
F__inference_dropout_15_layer_call_and_return_conditional_losses_109554

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿmm@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_110702

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿK:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs
½
M
1__inference_max_pooling2d_11_layer_call_fn_110955

inputs
identityÝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_109381
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

¤
(__inference_model_5_layer_call_fn_110528

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
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109468p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

þ
E__inference_conv2d_11_layer_call_and_return_conditional_losses_109436

inputs8
conv2d_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
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
:ÿÿÿÿÿÿÿÿÿ  `X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ$$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@
 
_user_specified_nameinputs
ë
í
(__inference_model_3_layer_call_fn_110056
input_16
input_17!
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

unknown_11:	K

unknown_12:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_16input_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_109991o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_16:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_17
ù
d
F__inference_dropout_15_layer_call_and_return_conditional_losses_109422

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿmm@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
 
_user_specified_nameinputs
Ý*

C__inference_model_5_layer_call_and_return_conditional_losses_109728
input_18*
conv2d_10_109694:@
conv2d_10_109696:@+
batch_normalization_10_109699:@+
batch_normalization_10_109701:@+
batch_normalization_10_109703:@+
batch_normalization_10_109705:@*
conv2d_11_109710:@`
conv2d_11_109712:`+
batch_normalization_11_109715:`+
batch_normalization_11_109717:`+
batch_normalization_11_109719:`+
batch_normalization_11_109721:`
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_18conv2d_10_109694conv2d_10_109696*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_109402
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_109699batch_normalization_10_109701batch_normalization_10_109703batch_normalization_10_109705*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_109254ø
dropout_15/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_109422ð
 max_pooling2d_10/PartitionedCallPartitionedCall#dropout_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_109305¢
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_109710conv2d_11_109712*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_109436
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_109715batch_normalization_11_109717batch_normalization_11_109719batch_normalization_11_109721*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_109330ø
dropout_16/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_109456ð
 max_pooling2d_11/PartitionedCallPartitionedCall#dropout_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_109381á
flatten_5/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_109465r
IdentityIdentity"flatten_5/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKð
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_18
Ý
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_109819

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿK:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs
È¶
­
"__inference__traced_restore_111281
file_prefix2
assignvariableop_dense_5_kernel:	K-
assignvariableop_1_dense_5_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: =
#assignvariableop_6_conv2d_10_kernel:@/
!assignvariableop_7_conv2d_10_bias:@=
/assignvariableop_8_batch_normalization_10_gamma:@<
.assignvariableop_9_batch_normalization_10_beta:@D
6assignvariableop_10_batch_normalization_10_moving_mean:@H
:assignvariableop_11_batch_normalization_10_moving_variance:@>
$assignvariableop_12_conv2d_11_kernel:@`0
"assignvariableop_13_conv2d_11_bias:`>
0assignvariableop_14_batch_normalization_11_gamma:`=
/assignvariableop_15_batch_normalization_11_beta:`D
6assignvariableop_16_batch_normalization_11_moving_mean:`H
:assignvariableop_17_batch_normalization_11_moving_variance:`#
assignvariableop_18_total: #
assignvariableop_19_count: %
assignvariableop_20_total_1: %
assignvariableop_21_count_1: ,
"assignvariableop_22_true_positives: -
#assignvariableop_23_false_positives: -
#assignvariableop_24_false_negatives: 2
(assignvariableop_25_weights_intermediate: <
)assignvariableop_26_adam_dense_5_kernel_m:	K5
'assignvariableop_27_adam_dense_5_bias_m:E
+assignvariableop_28_adam_conv2d_10_kernel_m:@7
)assignvariableop_29_adam_conv2d_10_bias_m:@E
7assignvariableop_30_adam_batch_normalization_10_gamma_m:@D
6assignvariableop_31_adam_batch_normalization_10_beta_m:@E
+assignvariableop_32_adam_conv2d_11_kernel_m:@`7
)assignvariableop_33_adam_conv2d_11_bias_m:`E
7assignvariableop_34_adam_batch_normalization_11_gamma_m:`D
6assignvariableop_35_adam_batch_normalization_11_beta_m:`<
)assignvariableop_36_adam_dense_5_kernel_v:	K5
'assignvariableop_37_adam_dense_5_bias_v:E
+assignvariableop_38_adam_conv2d_10_kernel_v:@7
)assignvariableop_39_adam_conv2d_10_bias_v:@E
7assignvariableop_40_adam_batch_normalization_10_gamma_v:@D
6assignvariableop_41_adam_batch_normalization_10_beta_v:@E
+assignvariableop_42_adam_conv2d_11_kernel_v:@`7
)assignvariableop_43_adam_conv2d_11_bias_v:`E
7assignvariableop_44_adam_batch_normalization_11_gamma_v:`D
6assignvariableop_45_adam_batch_normalization_11_beta_v:`
identity_47¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ö
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*
valueB/B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/2/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ò
_output_shapes¿
¼:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_10_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_10_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_10_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_10_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_10_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_10_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_11_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_11_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_11_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_11_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_11_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_11_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_true_positivesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp#assignvariableop_23_false_positivesIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp#assignvariableop_24_false_negativesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp(assignvariableop_25_weights_intermediateIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_5_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_5_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_conv2d_10_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_conv2d_10_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_30AssignVariableOp7assignvariableop_30_adam_batch_normalization_10_gamma_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_batch_normalization_10_beta_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_conv2d_11_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_conv2d_11_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_batch_normalization_11_gamma_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_11_beta_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_5_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_5_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp+assignvariableop_38_adam_conv2d_10_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_conv2d_10_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adam_batch_normalization_10_gamma_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_10_beta_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp+assignvariableop_42_adam_conv2d_11_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_conv2d_11_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adam_batch_normalization_11_gamma_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_batch_normalization_11_beta_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ã
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_47IdentityIdentity_46:output:0^NoOp_1*
T0*
_output_shapes
: °
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
ù
d
F__inference_dropout_15_layer_call_and_return_conditional_losses_110819

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿmm@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
 
_user_specified_nameinputs
	
Ò
7__inference_batch_normalization_11_layer_call_fn_110874

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_109330
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ç
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_110971

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ%  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

`:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`
 
_user_specified_nameinputs

¦
(__inference_model_5_layer_call_fn_109495
input_18!
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
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinput_18unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_109468p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
input_18
ü>
®
C__inference_model_5_layer_call_and_return_conditional_losses_110609

inputsB
(conv2d_10_conv2d_readvariableop_resource:@7
)conv2d_10_biasadd_readvariableop_resource:@<
.batch_normalization_10_readvariableop_resource:@>
0batch_normalization_10_readvariableop_1_resource:@M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_11_conv2d_readvariableop_resource:@`7
)conv2d_11_biasadd_readvariableop_resource:`<
.batch_normalization_11_readvariableop_resource:`>
0batch_normalization_11_readvariableop_1_resource:`M
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:`O
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:`
identity¢6batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_10/ReadVariableOp¢'batch_normalization_10/ReadVariableOp_1¢6batch_normalization_11/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_11/ReadVariableOp¢'batch_normalization_11/ReadVariableOp_1¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp¢ conv2d_11/BiasAdd/ReadVariableOp¢conv2d_11/Conv2D/ReadVariableOp
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0®
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
paddingVALID*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@l
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¿
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿmm@:@:@:@:@:*
epsilon%o:*
is_training( 
dropout_15/IdentityIdentity+batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@®
max_pooling2d_10/MaxPoolMaxPooldropout_15/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@*
ksize
*
paddingVALID*
strides

conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0É
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
paddingVALID*
strides

 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `l
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:`*
dtype0
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:`*
dtype0²
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0¶
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0¿
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  `:`:`:`:`:*
epsilon%o:*
is_training( 
dropout_16/IdentityIdentity+batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `®
max_pooling2d_11/MaxPoolMaxPooldropout_16/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`*
ksize
*
paddingVALID*
strides
`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ%  
flatten_5/ReshapeReshape!max_pooling2d_11/MaxPool:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKj
IdentityIdentityflatten_5/Reshape:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÜ
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
×*

C__inference_model_5_layer_call_and_return_conditional_losses_109468

inputs*
conv2d_10_109403:@
conv2d_10_109405:@+
batch_normalization_10_109408:@+
batch_normalization_10_109410:@+
batch_normalization_10_109412:@+
batch_normalization_10_109414:@*
conv2d_11_109437:@`
conv2d_11_109439:`+
batch_normalization_11_109442:`+
batch_normalization_11_109444:`+
batch_normalization_11_109446:`+
batch_normalization_11_109448:`
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCallÿ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_109403conv2d_10_109405*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_109402
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_109408batch_normalization_10_109410batch_normalization_10_109412batch_normalization_10_109414*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_109254ø
dropout_15/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_109422ð
 max_pooling2d_10/PartitionedCallPartitionedCall#dropout_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_109305¢
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_109437conv2d_11_109439*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_109436
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_109442batch_normalization_11_109444batch_normalization_11_109446batch_normalization_11_109448*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_109330ø
dropout_16/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_109456ð
 max_pooling2d_11/PartitionedCallPartitionedCall#dropout_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_109381á
flatten_5/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_109465r
IdentityIdentity"flatten_5/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKð
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
Í
»
C__inference_model_3_layer_call_and_return_conditional_losses_110329
inputs_0
inputs_1J
0model_5_conv2d_10_conv2d_readvariableop_resource:@?
1model_5_conv2d_10_biasadd_readvariableop_resource:@D
6model_5_batch_normalization_10_readvariableop_resource:@F
8model_5_batch_normalization_10_readvariableop_1_resource:@U
Gmodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:@W
Imodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:@J
0model_5_conv2d_11_conv2d_readvariableop_resource:@`?
1model_5_conv2d_11_biasadd_readvariableop_resource:`D
6model_5_batch_normalization_11_readvariableop_resource:`F
8model_5_batch_normalization_11_readvariableop_1_resource:`U
Gmodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:`W
Imodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:`9
&dense_5_matmul_readvariableop_resource:	K5
'dense_5_biasadd_readvariableop_resource:
identity¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢>model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp¢@model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¢@model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp¢Bmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1¢-model_5/batch_normalization_10/ReadVariableOp¢/model_5/batch_normalization_10/ReadVariableOp_1¢/model_5/batch_normalization_10/ReadVariableOp_2¢/model_5/batch_normalization_10/ReadVariableOp_3¢>model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp¢@model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1¢@model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp¢Bmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1¢-model_5/batch_normalization_11/ReadVariableOp¢/model_5/batch_normalization_11/ReadVariableOp_1¢/model_5/batch_normalization_11/ReadVariableOp_2¢/model_5/batch_normalization_11/ReadVariableOp_3¢(model_5/conv2d_10/BiasAdd/ReadVariableOp¢*model_5/conv2d_10/BiasAdd_1/ReadVariableOp¢'model_5/conv2d_10/Conv2D/ReadVariableOp¢)model_5/conv2d_10/Conv2D_1/ReadVariableOp¢(model_5/conv2d_11/BiasAdd/ReadVariableOp¢*model_5/conv2d_11/BiasAdd_1/ReadVariableOp¢'model_5/conv2d_11/Conv2D/ReadVariableOp¢)model_5/conv2d_11/Conv2D_1/ReadVariableOp 
'model_5/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_5_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0À
model_5/conv2d_10/Conv2DConv2Dinputs_0/model_5/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
paddingVALID*
strides

(model_5/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
model_5/conv2d_10/BiasAddBiasAdd!model_5/conv2d_10/Conv2D:output:00model_5/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@|
model_5/conv2d_10/ReluRelu"model_5/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@ 
-model_5/batch_normalization_10/ReadVariableOpReadVariableOp6model_5_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0¤
/model_5/batch_normalization_10/ReadVariableOp_1ReadVariableOp8model_5_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0Â
>model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Æ
@model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ï
/model_5/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3$model_5/conv2d_10/Relu:activations:05model_5/batch_normalization_10/ReadVariableOp:value:07model_5/batch_normalization_10/ReadVariableOp_1:value:0Fmodel_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿmm@:@:@:@:@:*
epsilon%o:*
is_training( 
model_5/dropout_15/IdentityIdentity3model_5/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@¾
 model_5/max_pooling2d_10/MaxPoolMaxPool$model_5/dropout_15/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@*
ksize
*
paddingVALID*
strides
 
'model_5/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_5_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0á
model_5/conv2d_11/Conv2DConv2D)model_5/max_pooling2d_10/MaxPool:output:0/model_5/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
paddingVALID*
strides

(model_5/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0³
model_5/conv2d_11/BiasAddBiasAdd!model_5/conv2d_11/Conv2D:output:00model_5/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `|
model_5/conv2d_11/ReluRelu"model_5/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ` 
-model_5/batch_normalization_11/ReadVariableOpReadVariableOp6model_5_batch_normalization_11_readvariableop_resource*
_output_shapes
:`*
dtype0¤
/model_5/batch_normalization_11/ReadVariableOp_1ReadVariableOp8model_5_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:`*
dtype0Â
>model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0Æ
@model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ï
/model_5/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3$model_5/conv2d_11/Relu:activations:05model_5/batch_normalization_11/ReadVariableOp:value:07model_5/batch_normalization_11/ReadVariableOp_1:value:0Fmodel_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  `:`:`:`:`:*
epsilon%o:*
is_training( 
model_5/dropout_16/IdentityIdentity3model_5/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `¾
 model_5/max_pooling2d_11/MaxPoolMaxPool$model_5/dropout_16/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`*
ksize
*
paddingVALID*
strides
h
model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ%  ¤
model_5/flatten_5/ReshapeReshape)model_5/max_pooling2d_11/MaxPool:output:0 model_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK¢
)model_5/conv2d_10/Conv2D_1/ReadVariableOpReadVariableOp0model_5_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ä
model_5/conv2d_10/Conv2D_1Conv2Dinputs_11model_5/conv2d_10/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@*
paddingVALID*
strides

*model_5/conv2d_10/BiasAdd_1/ReadVariableOpReadVariableOp1model_5_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¹
model_5/conv2d_10/BiasAdd_1BiasAdd#model_5/conv2d_10/Conv2D_1:output:02model_5/conv2d_10/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@
model_5/conv2d_10/Relu_1Relu$model_5/conv2d_10/BiasAdd_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@¢
/model_5/batch_normalization_10/ReadVariableOp_2ReadVariableOp6model_5_batch_normalization_10_readvariableop_resource*
_output_shapes
:@*
dtype0¤
/model_5/batch_normalization_10/ReadVariableOp_3ReadVariableOp8model_5_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ä
@model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGmodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0È
Bmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpImodel_5_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ù
1model_5/batch_normalization_10/FusedBatchNormV3_1FusedBatchNormV3&model_5/conv2d_10/Relu_1:activations:07model_5/batch_normalization_10/ReadVariableOp_2:value:07model_5/batch_normalization_10/ReadVariableOp_3:value:0Hmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp:value:0Jmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿmm@:@:@:@:@:*
epsilon%o:*
is_training( 
model_5/dropout_15/Identity_1Identity5model_5/batch_normalization_10/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿmm@Â
"model_5/max_pooling2d_10/MaxPool_1MaxPool&model_5/dropout_15/Identity_1:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$$@*
ksize
*
paddingVALID*
strides
¢
)model_5/conv2d_11/Conv2D_1/ReadVariableOpReadVariableOp0model_5_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0ç
model_5/conv2d_11/Conv2D_1Conv2D+model_5/max_pooling2d_10/MaxPool_1:output:01model_5/conv2d_11/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `*
paddingVALID*
strides

*model_5/conv2d_11/BiasAdd_1/ReadVariableOpReadVariableOp1model_5_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0¹
model_5/conv2d_11/BiasAdd_1BiasAdd#model_5/conv2d_11/Conv2D_1:output:02model_5/conv2d_11/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
model_5/conv2d_11/Relu_1Relu$model_5/conv2d_11/BiasAdd_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `¢
/model_5/batch_normalization_11/ReadVariableOp_2ReadVariableOp6model_5_batch_normalization_11_readvariableop_resource*
_output_shapes
:`*
dtype0¤
/model_5/batch_normalization_11/ReadVariableOp_3ReadVariableOp8model_5_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ä
@model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOpReadVariableOpGmodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0È
Bmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpImodel_5_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ù
1model_5/batch_normalization_11/FusedBatchNormV3_1FusedBatchNormV3&model_5/conv2d_11/Relu_1:activations:07model_5/batch_normalization_11/ReadVariableOp_2:value:07model_5/batch_normalization_11/ReadVariableOp_3:value:0Hmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp:value:0Jmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ  `:`:`:`:`:*
epsilon%o:*
is_training( 
model_5/dropout_16/Identity_1Identity5model_5/batch_normalization_11/FusedBatchNormV3_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `Â
"model_5/max_pooling2d_11/MaxPool_1MaxPool&model_5/dropout_16/Identity_1:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`*
ksize
*
paddingVALID*
strides
j
model_5/flatten_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ%  ª
model_5/flatten_5/Reshape_1Reshape+model_5/max_pooling2d_11/MaxPool_1:output:0"model_5/flatten_5/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
tf.math.subtract_5/SubSub"model_5/flatten_5/Reshape:output:0$model_5/flatten_5/Reshape_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKg
tf.math.abs_5/AbsAbstf.math.subtract_5/Sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿKi
dropout_17/IdentityIdentitytf.math.abs_5/Abs:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	K*
dtype0
dense_5/MatMulMatMuldropout_17/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp?^model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOpA^model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1A^model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOpC^model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1.^model_5/batch_normalization_10/ReadVariableOp0^model_5/batch_normalization_10/ReadVariableOp_10^model_5/batch_normalization_10/ReadVariableOp_20^model_5/batch_normalization_10/ReadVariableOp_3?^model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOpA^model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1A^model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOpC^model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1.^model_5/batch_normalization_11/ReadVariableOp0^model_5/batch_normalization_11/ReadVariableOp_10^model_5/batch_normalization_11/ReadVariableOp_20^model_5/batch_normalization_11/ReadVariableOp_3)^model_5/conv2d_10/BiasAdd/ReadVariableOp+^model_5/conv2d_10/BiasAdd_1/ReadVariableOp(^model_5/conv2d_10/Conv2D/ReadVariableOp*^model_5/conv2d_10/Conv2D_1/ReadVariableOp)^model_5/conv2d_11/BiasAdd/ReadVariableOp+^model_5/conv2d_11/BiasAdd_1/ReadVariableOp(^model_5/conv2d_11/Conv2D/ReadVariableOp*^model_5/conv2d_11/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿàà:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2
>model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp>model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2
@model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1@model_5/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12
@model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp@model_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp2
Bmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_1Bmodel_5/batch_normalization_10/FusedBatchNormV3_1/ReadVariableOp_12^
-model_5/batch_normalization_10/ReadVariableOp-model_5/batch_normalization_10/ReadVariableOp2b
/model_5/batch_normalization_10/ReadVariableOp_1/model_5/batch_normalization_10/ReadVariableOp_12b
/model_5/batch_normalization_10/ReadVariableOp_2/model_5/batch_normalization_10/ReadVariableOp_22b
/model_5/batch_normalization_10/ReadVariableOp_3/model_5/batch_normalization_10/ReadVariableOp_32
>model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp>model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2
@model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1@model_5/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12
@model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp@model_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp2
Bmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_1Bmodel_5/batch_normalization_11/FusedBatchNormV3_1/ReadVariableOp_12^
-model_5/batch_normalization_11/ReadVariableOp-model_5/batch_normalization_11/ReadVariableOp2b
/model_5/batch_normalization_11/ReadVariableOp_1/model_5/batch_normalization_11/ReadVariableOp_12b
/model_5/batch_normalization_11/ReadVariableOp_2/model_5/batch_normalization_11/ReadVariableOp_22b
/model_5/batch_normalization_11/ReadVariableOp_3/model_5/batch_normalization_11/ReadVariableOp_32T
(model_5/conv2d_10/BiasAdd/ReadVariableOp(model_5/conv2d_10/BiasAdd/ReadVariableOp2X
*model_5/conv2d_10/BiasAdd_1/ReadVariableOp*model_5/conv2d_10/BiasAdd_1/ReadVariableOp2R
'model_5/conv2d_10/Conv2D/ReadVariableOp'model_5/conv2d_10/Conv2D/ReadVariableOp2V
)model_5/conv2d_10/Conv2D_1/ReadVariableOp)model_5/conv2d_10/Conv2D_1/ReadVariableOp2T
(model_5/conv2d_11/BiasAdd/ReadVariableOp(model_5/conv2d_11/BiasAdd/ReadVariableOp2X
*model_5/conv2d_11/BiasAdd_1/ReadVariableOp*model_5/conv2d_11/BiasAdd_1/ReadVariableOp2R
'model_5/conv2d_11/Conv2D/ReadVariableOp'model_5/conv2d_11/Conv2D/ReadVariableOp2V
)model_5/conv2d_11/Conv2D_1/ReadVariableOp)model_5/conv2d_11/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
"
_user_specified_name
inputs/1"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ÿ
serving_defaultë
G
input_16;
serving_default_input_16:0ÿÿÿÿÿÿÿÿÿàà
G
input_17;
serving_default_input_17:0ÿÿÿÿÿÿÿÿÿàà;
dense_50
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:øù
ò
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

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
¼
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'_random_generator
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
»

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer

2iter

3beta_1

4beta_2
	5decay*mÑ+mÒ6mÓ7mÔ8mÕ9mÖ<m×=mØ>mÙ?mÚ*vÛ+vÜ6vÝ7vÞ8vß9và<vá=vâ>vã?vä"
	optimizer

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
Ê
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
î2ë
(__inference_model_3_layer_call_fn_109870
(__inference_model_3_layer_call_fn_110196
(__inference_model_3_layer_call_fn_110230
(__inference_model_3_layer_call_fn_110056À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
C__inference_model_3_layer_call_and_return_conditional_losses_110329
C__inference_model_3_layer_call_and_return_conditional_losses_110463
C__inference_model_3_layer_call_and_return_conditional_losses_110107
C__inference_model_3_layer_call_and_return_conditional_losses_110158À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
×BÔ
!__inference__wrapped_model_109232input_16input_17"
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
,
Gserving_default"
signature_map
"
_tf_keras_input_layer
»

6kernel
7bias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
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
¼
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y_random_generator
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
»

<kernel
=bias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
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
¼
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s_random_generator
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
§
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
î2ë
(__inference_model_5_layer_call_fn_109495
(__inference_model_5_layer_call_fn_110528
(__inference_model_5_layer_call_fn_110557
(__inference_model_5_layer_call_fn_109691À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
C__inference_model_5_layer_call_and_return_conditional_losses_110609
C__inference_model_5_layer_call_and_return_conditional_losses_110675
C__inference_model_5_layer_call_and_return_conditional_losses_109728
C__inference_model_5_layer_call_and_return_conditional_losses_109765À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_17_layer_call_fn_110680
+__inference_dropout_17_layer_call_fn_110685´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_17_layer_call_and_return_conditional_losses_110690
F__inference_dropout_17_layer_call_and_return_conditional_losses_110702´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!:	K2dense_5/kernel
:2dense_5/bias
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_5_layer_call_fn_110711¢
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
í2ê
C__inference_dense_5_layer_call_and_return_conditional_losses_110722¢
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
*:(@2conv2d_10/kernel
:@2conv2d_10/bias
*:(@2batch_normalization_10/gamma
):'@2batch_normalization_10/beta
2:0@ (2"batch_normalization_10/moving_mean
6:4@ (2&batch_normalization_10/moving_variance
*:(@`2conv2d_11/kernel
:`2conv2d_11/bias
*:(`2batch_normalization_11/gamma
):'`2batch_normalization_11/beta
2:0` (2"batch_normalization_11/moving_mean
6:4` (2&batch_normalization_11/moving_variance
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
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÔBÑ
$__inference_signature_wrapper_110499input_16input_17"
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
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_conv2d_10_layer_call_fn_110731¢
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
E__inference_conv2d_10_layer_call_and_return_conditional_losses_110742¢
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
¬2©
7__inference_batch_normalization_10_layer_call_fn_110755
7__inference_batch_normalization_10_layer_call_fn_110768´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_110786
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_110804´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_15_layer_call_fn_110809
+__inference_dropout_15_layer_call_fn_110814´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_15_layer_call_and_return_conditional_losses_110819
F__inference_dropout_15_layer_call_and_return_conditional_losses_110831´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_max_pooling2d_10_layer_call_fn_110836¢
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
ö2ó
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_110841¢
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
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_conv2d_11_layer_call_fn_110850¢
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
E__inference_conv2d_11_layer_call_and_return_conditional_losses_110861¢
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
²
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
¬2©
7__inference_batch_normalization_11_layer_call_fn_110874
7__inference_batch_normalization_11_layer_call_fn_110887´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_110905
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_110923´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
o	variables
ptrainable_variables
qregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_16_layer_call_fn_110928
+__inference_dropout_16_layer_call_fn_110933´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_16_layer_call_and_return_conditional_losses_110938
F__inference_dropout_16_layer_call_and_return_conditional_losses_110950´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_max_pooling2d_11_layer_call_fn_110955¢
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
ö2ó
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_110960¢
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
µ
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_flatten_5_layer_call_fn_110965¢
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
E__inference_flatten_5_layer_call_and_return_conditional_losses_110971¢
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

Átotal

Âcount
Ã	variables
Ä	keras_api"
_tf_keras_metric
c

Åtotal

Æcount
Ç
_fn_kwargs
È	variables
É	keras_api"
_tf_keras_metric
§
Ê
init_shape
Ëtrue_positives
Ìfalse_positives
Ífalse_negatives
Îweights_intermediate
Ï	variables
Ð	keras_api"
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
Á0
Â1"
trackable_list_wrapper
.
Ã	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Å0
Æ1"
trackable_list_wrapper
.
È	variables"
_generic_user_object
 "
trackable_list_wrapper
:  (2true_positives
:  (2false_positives
:  (2false_negatives
 :  (2weights_intermediate
@
Ë0
Ì1
Í2
Î3"
trackable_list_wrapper
.
Ï	variables"
_generic_user_object
&:$	K2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
/:-@2Adam/conv2d_10/kernel/m
!:@2Adam/conv2d_10/bias/m
/:-@2#Adam/batch_normalization_10/gamma/m
.:,@2"Adam/batch_normalization_10/beta/m
/:-@`2Adam/conv2d_11/kernel/m
!:`2Adam/conv2d_11/bias/m
/:-`2#Adam/batch_normalization_11/gamma/m
.:,`2"Adam/batch_normalization_11/beta/m
&:$	K2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
/:-@2Adam/conv2d_10/kernel/v
!:@2Adam/conv2d_10/bias/v
/:-@2#Adam/batch_normalization_10/gamma/v
.:,@2"Adam/batch_normalization_10/beta/v
/:-@`2Adam/conv2d_11/kernel/v
!:`2Adam/conv2d_11/bias/v
/:-`2#Adam/batch_normalization_11/gamma/v
.:,`2"Adam/batch_normalization_11/beta/vÙ
!__inference__wrapped_model_109232³6789:;<=>?@A*+n¢k
d¢a
_\
,)
input_16ÿÿÿÿÿÿÿÿÿàà
,)
input_17ÿÿÿÿÿÿÿÿÿàà
ª "1ª.
,
dense_5!
dense_5ÿÿÿÿÿÿÿÿÿí
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_11078689:;M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 í
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_11080489:;M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Å
7__inference_batch_normalization_10_layer_call_fn_11075589:;M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Å
7__inference_batch_normalization_10_layer_call_fn_11076889:;M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@í
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_110905>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 í
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_110923>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 Å
7__inference_batch_normalization_11_layer_call_fn_110874>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`Å
7__inference_batch_normalization_11_layer_call_fn_110887>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`·
E__inference_conv2d_10_layer_call_and_return_conditional_losses_110742n679¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿmm@
 
*__inference_conv2d_10_layer_call_fn_110731a679¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª " ÿÿÿÿÿÿÿÿÿmm@µ
E__inference_conv2d_11_layer_call_and_return_conditional_losses_110861l<=7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ$$@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  `
 
*__inference_conv2d_11_layer_call_fn_110850_<=7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ$$@
ª " ÿÿÿÿÿÿÿÿÿ  `¤
C__inference_dense_5_layer_call_and_return_conditional_losses_110722]*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿK
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_5_layer_call_fn_110711P*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿK
ª "ÿÿÿÿÿÿÿÿÿ¶
F__inference_dropout_15_layer_call_and_return_conditional_losses_110819l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿmm@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿmm@
 ¶
F__inference_dropout_15_layer_call_and_return_conditional_losses_110831l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿmm@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿmm@
 
+__inference_dropout_15_layer_call_fn_110809_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿmm@
p 
ª " ÿÿÿÿÿÿÿÿÿmm@
+__inference_dropout_15_layer_call_fn_110814_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿmm@
p
ª " ÿÿÿÿÿÿÿÿÿmm@¶
F__inference_dropout_16_layer_call_and_return_conditional_losses_110938l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  `
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  `
 ¶
F__inference_dropout_16_layer_call_and_return_conditional_losses_110950l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  `
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  `
 
+__inference_dropout_16_layer_call_fn_110928_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  `
p 
ª " ÿÿÿÿÿÿÿÿÿ  `
+__inference_dropout_16_layer_call_fn_110933_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  `
p
ª " ÿÿÿÿÿÿÿÿÿ  `¨
F__inference_dropout_17_layer_call_and_return_conditional_losses_110690^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿK
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿK
 ¨
F__inference_dropout_17_layer_call_and_return_conditional_losses_110702^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿK
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿK
 
+__inference_dropout_17_layer_call_fn_110680Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿK
p 
ª "ÿÿÿÿÿÿÿÿÿK
+__inference_dropout_17_layer_call_fn_110685Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿK
p
ª "ÿÿÿÿÿÿÿÿÿKª
E__inference_flatten_5_layer_call_and_return_conditional_losses_110971a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

`
ª "&¢#

0ÿÿÿÿÿÿÿÿÿK
 
*__inference_flatten_5_layer_call_fn_110965T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

`
ª "ÿÿÿÿÿÿÿÿÿKï
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_110841R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_max_pooling2d_10_layer_call_fn_110836R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_110960R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_max_pooling2d_11_layer_call_fn_110955R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ÷
C__inference_model_3_layer_call_and_return_conditional_losses_110107¯6789:;<=>?@A*+v¢s
l¢i
_\
,)
input_16ÿÿÿÿÿÿÿÿÿàà
,)
input_17ÿÿÿÿÿÿÿÿÿàà
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ÷
C__inference_model_3_layer_call_and_return_conditional_losses_110158¯6789:;<=>?@A*+v¢s
l¢i
_\
,)
input_16ÿÿÿÿÿÿÿÿÿàà
,)
input_17ÿÿÿÿÿÿÿÿÿàà
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ÷
C__inference_model_3_layer_call_and_return_conditional_losses_110329¯6789:;<=>?@A*+v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿàà
,)
inputs/1ÿÿÿÿÿÿÿÿÿàà
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ÷
C__inference_model_3_layer_call_and_return_conditional_losses_110463¯6789:;<=>?@A*+v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿàà
,)
inputs/1ÿÿÿÿÿÿÿÿÿàà
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
(__inference_model_3_layer_call_fn_109870¢6789:;<=>?@A*+v¢s
l¢i
_\
,)
input_16ÿÿÿÿÿÿÿÿÿàà
,)
input_17ÿÿÿÿÿÿÿÿÿàà
p 

 
ª "ÿÿÿÿÿÿÿÿÿÏ
(__inference_model_3_layer_call_fn_110056¢6789:;<=>?@A*+v¢s
l¢i
_\
,)
input_16ÿÿÿÿÿÿÿÿÿàà
,)
input_17ÿÿÿÿÿÿÿÿÿàà
p

 
ª "ÿÿÿÿÿÿÿÿÿÏ
(__inference_model_3_layer_call_fn_110196¢6789:;<=>?@A*+v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿàà
,)
inputs/1ÿÿÿÿÿÿÿÿÿàà
p 

 
ª "ÿÿÿÿÿÿÿÿÿÏ
(__inference_model_3_layer_call_fn_110230¢6789:;<=>?@A*+v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿàà
,)
inputs/1ÿÿÿÿÿÿÿÿÿàà
p

 
ª "ÿÿÿÿÿÿÿÿÿÂ
C__inference_model_5_layer_call_and_return_conditional_losses_109728{6789:;<=>?@AC¢@
9¢6
,)
input_18ÿÿÿÿÿÿÿÿÿàà
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿK
 Â
C__inference_model_5_layer_call_and_return_conditional_losses_109765{6789:;<=>?@AC¢@
9¢6
,)
input_18ÿÿÿÿÿÿÿÿÿàà
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿK
 À
C__inference_model_5_layer_call_and_return_conditional_losses_110609y6789:;<=>?@AA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿK
 À
C__inference_model_5_layer_call_and_return_conditional_losses_110675y6789:;<=>?@AA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿK
 
(__inference_model_5_layer_call_fn_109495n6789:;<=>?@AC¢@
9¢6
,)
input_18ÿÿÿÿÿÿÿÿÿàà
p 

 
ª "ÿÿÿÿÿÿÿÿÿK
(__inference_model_5_layer_call_fn_109691n6789:;<=>?@AC¢@
9¢6
,)
input_18ÿÿÿÿÿÿÿÿÿàà
p

 
ª "ÿÿÿÿÿÿÿÿÿK
(__inference_model_5_layer_call_fn_110528l6789:;<=>?@AA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "ÿÿÿÿÿÿÿÿÿK
(__inference_model_5_layer_call_fn_110557l6789:;<=>?@AA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "ÿÿÿÿÿÿÿÿÿKð
$__inference_signature_wrapper_110499Ç6789:;<=>?@A*+¢~
¢ 
wªt
8
input_16,)
input_16ÿÿÿÿÿÿÿÿÿàà
8
input_17,)
input_17ÿÿÿÿÿÿÿÿÿàà"1ª.
,
dense_5!
dense_5ÿÿÿÿÿÿÿÿÿ