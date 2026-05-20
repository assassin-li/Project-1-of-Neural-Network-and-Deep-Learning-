# Convolutional Neural Network & Backpropagation Algorithm

Yunuo Cai

## Content

- (Recall) Back Propagation(BP) Algorithm on Neural Network

- (Recall) Convolutional Network(CNN)

- BP Algorithm on CNN

## Content

- (Recall) Back Propagation(BP) Algorithm on Neural Network

- (Recall) Convolutional Network(CNN)

- BP Algorithm on CNN

## How to Optimize a Network h?

$$
\underset { \theta } { \mathrm { a r g m i n } } \ J ( \mathsf { h } ( \theta \mid X ) , \ y )
$$

- θ : Parameters

- X : Training Data (Fixed)

- J : A Loss function

Gradient Descent:

$$
\theta _ { i + 1 } = \theta _ { i } - a \cdot \nabla _ { \theta } J ( \mathsf { h } ( \theta _ { i } \mid X ) , \ y ) \ \mathsf { f o r } \ i = 0 , 1 , 2 , \ldots
$$

Like a person walking down the hills.

<!-- image-->

## Gradient Descent

$$
\underset { \theta } { \mathrm { a r g m i n } } \ J ( \mathsf { h } ( \theta \mid X ) , \ y )
$$

- Compute $J ( \mathsf { h } ( \theta _ { i } \mid X ) , \ y )$ (Forward Propagation)

- Compute $\nabla _ { \theta } J ( \mathfrak { h } ( \theta _ { i } \mid X ) , \ y )$ (Backward Propagation)

- Update Parameters

Like a person walking down the hills.

<!-- image-->

## The Case in Neural Network - A Neuron

<!-- image-->

$$
h _ { W , b } ( x ) = f ( W x ) = f ( \sum _ { i = 1 } ^ { 3 } W _ { i } x _ { i } + b )
$$

where $f : \Re \mapsto \Re$ is called the activation function

## The Case in Neural Network 一 Multiple Neurons

<!-- image-->

Input Layer

Hidden Layer

Output Layer

## The Case in Neural Network 一 Forward Propagation

$n _ { l }$ : the number of layers

$L _ { l }$ : the layer 1

$\textit { W } _ { i j } ^ { ( l ) }$ : the weight associated with the connection between unit j in layer l, and unit i in layer $l + 1$

$b _ { i } ^ { ( l ) }$ : the bias associated with unit i inlayer $l + 1$

$a _ { \ i } ^ { \ ( l ) }$ : the acctivation unit i in layer l

$\boldsymbol { z } _ { i } ^ { \left( l \right) }$ : the total weighted sum of inputs to unit i in layer l

$$
a _ { 1 } ^ { ( 2 ) } = f ( W _ { 1 1 } ^ { ( 1 ) } x _ { 1 } + W _ { 1 2 } ^ { ( 1 ) } x _ { 2 } + W _ { 1 3 } ^ { ( 1 ) } x _ { 3 } + b _ { 1 } ^ { ( 1 ) } )
$$

$$
a _ { 2 } ^ { ( 2 ) } = f ( W _ { 2 1 } ^ { ( 1 ) } x _ { 1 } + W _ { 2 2 } ^ { ( 1 ) } x _ { 2 } + W _ { 2 3 } ^ { ( 1 ) } x _ { 3 } + b _ { 2 } ^ { ( 1 ) } )
$$

$$
a _ { 3 } ^ { ( 2 ) } = f ( W _ { 3 1 } ^ { ( 1 ) } x _ { 1 } + W _ { 3 2 } ^ { ( 1 ) } x _ { 2 } + W _ { 3 3 } ^ { ( 1 ) } x _ { 3 } + b _ { 3 } ^ { ( 1 ) } )
$$

$$
h _ { W , b } ( x ) = a _ { 1 } ^ { ( 3 ) } = f ( W _ { 1 1 } ^ { ( 2 ) } a _ { 1 } ^ { ( 2 ) } + W _ { 1 2 } ^ { ( 2 ) } a _ { 2 } ^ { ( 2 ) } + W _ { 1 3 } ^ { ( 2 ) } a _ { 3 } ^ { ( 2 ) } + b _ { 1 } ^ { ( 2 ) } )
$$

<!-- image-->  
Layer L  
LayerL2

## The Case in Neural Network 一 Forward Propagation

$n _ { l }$ : the number of layers

$L _ { l }$ : the layer 1

$\textit { W } _ { i j } ^ { ( l ) }$ : the weight associated with the connection between unit jin layer l, and unit i in layer $l + 1$

$b _ { i } ^ { ( l ) }$ : the bias associated with unit i inlayer $l + 1$

$a _ { \ i } ^ { \ ( l ) }$ : the acctivation unit i in layer l

$\boldsymbol { z } _ { i } ^ { \left( l \right) }$ : the total weighted sum of inputs to unit i in layer l

$$
a _ { 1 } ^ { ( 2 ) } = f ( W _ { 1 1 } ^ { ( 1 ) } x _ { 1 } + W _ { 1 2 } ^ { ( 1 ) } x _ { 2 } + W _ { 1 3 } ^ { ( 1 ) } x _ { 3 } + b _ { 1 } ^ { ( 1 ) } ) \ z _ { 1 } ^ { ( 2 ) }
$$

$$
\mathsf { a } _ { i } ^ { ( I ) } = f ( z _ { i } ^ { ( I ) } )
$$

$$
a _ { 2 } ^ { ( 2 ) } = f ( W _ { 2 1 } ^ { ( 1 ) } x _ { 1 } + W _ { 2 2 } ^ { ( 1 ) } x _ { 2 } + W _ { 2 3 } ^ { ( 1 ) } x _ { 3 } + b _ { 2 } ^ { ( 1 ) } ) \ z _ { 2 } ^ { ( 2 ) }
$$

$$
a _ { 3 } ^ { ( 2 ) } = f ( W _ { 3 1 } ^ { ( 1 ) } x _ { 1 } + W _ { 3 2 } ^ { ( 1 ) } x _ { 2 } + W _ { 3 3 } ^ { ( 1 ) } x _ { 3 } + b _ { 3 } ^ { ( 1 ) } ) \ z _ { 3 } ^ { ( 2 ) }
$$

<!-- image-->

$$
h _ { W , b } ( x ) = a _ { 1 } ^ { ( 3 ) } = f ( W _ { 1 1 } ^ { ( 2 ) } a _ { 1 } ^ { ( 2 ) } + W _ { 1 2 } ^ { ( 2 ) } a _ { 2 } ^ { ( 2 ) } + W _ { 1 3 } ^ { ( 2 ) } a _ { 3 } ^ { ( 2 ) } + b _ { 1 } ^ { ( 2 ) } )
$$

$$
z _ { \ l _ { 1 } } ^ { ( 3 ) }
$$

## The Case in Neural Network 一 Forward Propagation

$n _ { l }$ : the number of layers

$L _ { l }$ : the layer 1

$\textit { W } _ { i j } ^ { ( l ) }$ : the weight associated with the connection between unit jin layer l, and unit i in layer $l + 1$

$b _ { i } ^ { ( l ) }$ : the bias associated with unit i inlayer $l + 1$

$a _ { \ i } ^ { \ ( l ) }$ : the acctivation unit i in layer l

$\boldsymbol { z } _ { i } ^ { \left( l \right) }$ : the total weighted sum of inputs to unit i in layer l

LayerL

LayerL2

$$
z ^ { ( 2 ) } = W ^ { ( 1 ) } x + b ^ { ( 1 ) }
$$

$$
a ^ { ( 2 ) } = f ( z ^ { ( 2 ) } )
$$

<!-- image-->

$$
{ z ^ { ( 3 ) } = W ^ { ( 2 ) } a ^ { ( 2 ) } + b ^ { ( 2 ) } }
$$

<!-- image-->

$$
\boldsymbol { z } ^ { ( l + 1 ) } = \boldsymbol { W } ^ { ( l ) } \boldsymbol { a } ^ { ( l ) } + \boldsymbol { b } ^ { ( l ) }
$$

$$
h _ { W , b } ( x ) = a ^ { ( 3 ) } = f ( z ^ { ( 3 ) } )
$$

$$
a ^ { ( l + 1 ) } = f ( z ^ { ( l + 1 ) } )
$$

## The Case in Neural Network 一 Forward Propagation

· Loss Function:

$$
J ( W , b ; x , y ) = \frac { 1 } { 2 } \left\| h _ { W , b } ( x ) - y \right\| ^ { 2 } .
$$

<!-- image-->

$$
\begin{array} { c l c r } { { J ( W , b ) = \displaystyle \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } J ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } ) \right] + \frac { \lambda } { 2 } \sum _ { l = 1 } ^ { n _ { l } - 1 } \sum _ { i = 1 } ^ { s _ { l } } \sum _ { j = 1 } ^ { s _ { l + 1 } } \left( W _ { j i } ^ { ( l ) } \right) ^ { 2 } } } \\ { { = \displaystyle \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( \frac { 1 } { 2 } \left\| h _ { W , b } ( x ^ { ( i ) } ) - y ^ { ( i ) } \right\| ^ { 2 } \right) \right] + \frac { \lambda } { 2 } \sum _ { l = 1 } ^ { n _ { l } - 1 } \sum _ { i = 1 } ^ { s _ { l } } \sum _ { j = 1 } ^ { s _ { l + 1 } } \left( W _ { j i } ^ { ( l ) } \right) ^ { 2 } } } \end{array}
$$

## The Case in Neural Network 一 Backward Propagation

·Gradient Descent:

Regularization Term

$$
\begin{array} { c l c r } { \displaystyle { J ( W , b ) = \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } J ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } ) \right] + \left[ \frac { \lambda } { 2 } \sum _ { l = 1 } ^ { n _ { l } - 1 } \sum _ { i = 1 } ^ { s _ { l } } \sum _ { j = 1 } ^ { s _ { l + 1 } } \left( W _ { j i } ^ { ( l ) } \right) ^ { 2 } \right] } } \\ { \displaystyle { = \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( \frac { 1 } { 2 } \left\| h _ { W , b } ( x ^ { ( i ) } ) - y ^ { ( i ) } \right\| ^ { 2 } \right) \right] + \frac { \lambda } { 2 } \sum _ { l = 1 } ^ { n _ { l } - 1 } \sum _ { i = 1 } ^ { s _ { l } } \sum _ { j = 1 } ^ { s _ { l + 1 } } \left( W _ { j i } ^ { ( l ) } \right) ^ { 2 } } } \end{array}
$$

$$
\begin{array} { c } { { W _ { i j } ^ { ( l ) } = W _ { i j } ^ { ( l ) } - \alpha \displaystyle { \frac { \partial } { \partial W _ { i j } ^ { ( l ) } } } J ( W , b ) } } \\ { { b _ { i } ^ { ( l ) } = b _ { i } ^ { ( l ) } - \alpha \displaystyle { \frac { \partial } { \partial b _ { i } ^ { ( l ) } } } J ( W , b ) } } \end{array}
$$

<!-- image-->  
Layer L  
Layer L

## The Case in Neural Network - Chain Rule

·Compute: $\frac { \partial J } { \partial W _ { i j } ^ { ( I ) } }$

·Considering:

$$
\begin{array} { r l } { \bullet } & { \boldsymbol { a } _ { i } ^ { ( I + 1 ) } = \boldsymbol { f } ( \boldsymbol { z } _ { i } ^ { ( I + 1 ) } ) } \\ { \bullet } & { \boldsymbol { z } _ { i } ^ { ( I + 1 ) } = \sum _ { j = 1 } ^ { s _ { I } } \boldsymbol { W } _ { i j } ^ { ( I ) } \boldsymbol { a } _ { j } ^ { ( I ) } + \boldsymbol { b } _ { i } ^ { ( I ) } } \end{array}
$$

· Using Chain Rule:

· And we can also get:

<!-- image-->

$$
\begin{array} { r l r } {  { \frac { \partial J } { \partial W _ { i j } ^ { ( I ) } } = \frac { \partial J } { \partial z _ { i } ^ { ( I + 1 ) } } \cdot \frac { \partial z _ { i } ^ { ( I + 1 ) } } { \partial W _ { i j } ^ { ( I ) } } = \frac { \partial J } { \partial z _ { i } ^ { ( I + 1 ) } } \cdot a _ { j \underbrace { \mathrm { ~ w a t r i x } } _ { \mathrm { F o r m } } } ^ { ( I ) } } } \\ & { } & { \frac { \partial J } { \partial a _ { i } ^ { ( I ) } } = \displaystyle \sum _ { j } \frac { \partial J } { \partial z _ { j } ^ { ( I + 1 ) } } \cdot \frac { \partial z _ { j } ^ { ( I + 1 ) } } { \partial \mathsf { a } _ { i } ^ { ( I ) } } } \end{array}
$$

## The Case in Neural Network - A Modularized View

<!-- image-->

A Linear Layer consists of:

1. Parameters: W &b

2． Forward Function:

[INPUT] $\pmb { a } ^ { ( I ) }$

[OUTPUT] $\pmb { z } ^ { ( I + 1 ) } = \pmb { W } \pmb { a } ^ { ( I ) } + \pmb { b }$

[STORE] store $\pmb { a } ^ { ( I ) }$

3. Backward Function:

[INPUT] $\frac { \partial J } { \partial z ^ { ( I + 1 ) } }$

[OUTPUT] $\begin{array} { r } { \frac { \partial J } { \partial \pmb { a } ^ { ( I ) } } = \left( \frac { \partial \pmb { z } ^ { ( I + 1 ) } } { \partial \pmb { a } ^ { ( I ) } } \right) ^ { T } \frac { \partial J } { \partial \pmb { z } ^ { ( I + 1 ) } } = \pmb { W } ^ { T } \frac { \partial J } { \partial \pmb { z } ^ { ( I + 1 ) } } } \end{array}$

store $\begin{array} { r } { \frac { \partial J } { \partial W } = \pmb { a } ^ { ( I ) } \cdot \frac { \partial J } { \partial \pmb { z } ^ { ( I + 1 ) } } } \end{array}$ for later gradient descent

## General Cases - A Modularized View

<!-- image-->

A Layer consists of:

1. Some parameters for optimization(Optional)

2． Forward Function:

compute output based on given input and parameters store the input for computation in backward function

3. Backward Function:

[INPUT] derivatives to the output of forward function

[OUTPUT] derivatives to the input of forward function

[STORE] computes derivatives to the parameters(Optional)

## Intuition

· Gradients generates when a variable is involved in a computation:

$$
z _ { 1 } = w _ { 1 } a + b _ { 1 }
$$

$$
z _ { 2 } = w _ { 2 } a + b _ { 2 }
$$

·Another Case:

$$
\frac { \partial J } { \partial a } = \sum _ { j } \frac { \partial J } { \partial z _ { j } } \mathrm { ~ \cdot ~ } \frac { \partial z _ { j } } { \partial a }
$$

$$
z _ { 1 } = w a _ { 1 } + b
$$

$$
z _ { 2 } = w a _ { 2 } + b
$$

$$
\begin{array} { r } { \frac { \partial J } { \partial w } = \displaystyle \sum _ { j } \frac { \partial J } { \partial z _ { j } } \cdot \frac { \partial z _ { j } } { \partial w } } \\ { \frac { \partial J } { \partial b } = \displaystyle \sum _ { j } \frac { \partial J } { \partial z _ { j } } \cdot \frac { \partial z _ { j } } { \partial b } } \end{array}
$$

## Content

- (Recall) Back Propagation(BP) Algorithm on Neural Network

(Recall) Convolutional Network(CNN)

- BP Algorithm on CNN

## Convolutional Neural Network

FULLY CONNECTED NEURAL NET

<!-- image-->

<!-- image-->

## Convolutional Neural Network

<!-- image-->

<!-- image-->

A Convolution Kernel Working on RGB Image

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->