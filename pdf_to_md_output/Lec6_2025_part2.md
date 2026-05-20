## A Convolution Kernel Working on RGB Image

$$
C \times H \times W
$$

· Input Shape: $3 \times 3 2 \times 3 2$

· Kernel Shape: $3 \times 5 \times 5$

· Stride = 1, No Padding

· The Output Shape?

<!-- image-->

## A Convolution Kernel Working on RGB Image

$$
C \times H \times W
$$

· Input Shape: $3 \times 3 2 \times 3 2$

· Kernel Shape: $3 \times 5 \times 5$

· Stride = 1, No Padding

· The Output Shape?

$$
1 \times 2 8 \times 2 8
$$

<!-- image-->

$$
H _ { l + 1 } = \frac { ( H _ { l } + 2 \times p a d \_ h \_ \textmu _ { - } h \_ h ) } { s t r i d e \_ h } + 1 \qquad W _ { l + 1 } = \frac { ( W _ { l } + 2 \times p a d \_ w \_ w l \_ w ) } { s t r i d e \_ w } + 1
$$

## A Convolution Kernel Working on RGB Image

$$
C \times H \times W
$$

· Input Shape: $3 \times 3 2 \times 3 2$

· Kernel Shape: $3 \times 3 \times 3$

· Stride = 2, Padding = 1

· The Output Shape?

<!-- image-->

## A Convolution Kernel Working on RGB Image

$$
C \times H \times W
$$

· Input Shape: $3 \times 3 2 \times 3 2$

· Kernel Shape: $3 \times 3 \times 3$

· Stride = 2, Padding = 1

· The Output Shape?

1 × 15.5 × 15.5 ? (Flooring,or Special Padding)

<!-- image-->

Try it on your own using PyTorch or MindSpore.

$$
H _ { l + 1 } = \frac { ( H _ { l } + 2 \times p a d \_ h \_ \textmu _ { - } h \_ h ) } { s t r i d e \_ h } + 1 \qquad W _ { l + 1 } = \frac { ( W _ { l } + 2 \times p a d \_ w \_ w l \_ w ) } { s t r i d e \_ w } + 1
$$

## General Cases - Multiple Kernels

· Input Shape: $C _ { i n } \times H _ { i n } \times W _ { i n }$

· Kernel Shape: $C _ { o u t } \times C _ { i n } \times H \times W$

· Bias Shape: $\complement _ { o u t }$

· Stride S, Padding P

·Output Shape: $C _ { o u t } \times H _ { o u t } \times W _ { o u t }$

<!-- image-->

## Content

- (Recall) Back Propagation(BP) Algorithm on Neural Network

- (Recall) Convolutional Network(CNN)

- BP Algorithm on CNN

## One Operation Case

·Considering One Operation of One Kernel $y _ { \mathit { I o c a I } } = W * X _ { \mathit { I o c a I } } + b$ (逐元素乘后相加) This generates the following gradients:

$$
\begin{array} { r } { \frac { \partial { \cal J } } { \partial X _ { I o c a I } } = \frac { \partial { \cal J } } { \partial y _ { I o c a I } } \times \frac { \partial y _ { I o c a I } } { \partial X _ { I o c a I } } = \frac { \partial { \cal J } } { \partial y _ { I o c a I } } \times { \cal W } } \end{array}
$$

$$
\begin{array} { r } { \frac { \partial J } { \partial W } = \frac { \partial J } { \partial y _ { I o c a I } } \times \frac { \partial y _ { I o c a I } } { \partial W } = \frac { \partial J } { \partial y _ { I o c a I } } \times X _ { I o c a I } } \end{array}
$$

$$
\begin{array} { r } { \frac { \partial J } { \partial b } = \frac { \partial J } { \partial y _ { I o c a I } } \times \frac { \partial y _ { I o c a I } } { \partial \mathrm { b } } = \frac { \partial J } { \partial y _ { I o c a I } } } \end{array}
$$

<table><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>02</td><td rowspan=1 colspan=1>02</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>2 0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>00</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td></tr></table>

<table><tr><td></td><td rowspan=1 colspan=1>12.017.0</td><td rowspan=1 colspan=1>12.017.0</td></tr><tr><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>19.0</td></tr><tr><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>14.0</td></tr></table>

## One Operation Case

·Considering One Operation of One Kernel $y _ { \mathit { I o c a l } } = W * \ x _ { \mathit { I o c a l } } + b$ (逐元素乘后相加) This generates the following gradients:

$$
\begin{array} { r } { \frac { \partial { \cal J } } { \partial X _ { I o c a I } } = \frac { \partial { \cal J } } { \partial y _ { I o c a I } } \times \frac { \partial y _ { I o c a I } } { \partial X _ { I o c a I } } = \frac { \partial { \cal J } } { \partial y _ { I o c a I } } \times { \cal W } } \end{array}
$$

$$
\begin{array} { r } { \frac { \partial J } { \partial W } = \frac { \partial J } { \partial y _ { I o c a I } } \times \frac { \partial y _ { I o c a I } } { \partial W } = \frac { \partial J } { \partial y _ { I o c a I } } \times X _ { I o c a I } } \end{array}
$$

$$
\begin{array} { r } { \frac { \partial J } { \partial b } = \frac { \partial J } { \partial y _ { I o c a I } } \times \frac { \partial y _ { I o c a I } } { \partial \mathrm { b } } = \frac { \partial J } { \partial y _ { I o c a I } } } \end{array}
$$

<!-- image-->

<table><tr><td rowspan=1 colspan=3>ylocal</td></tr><tr><td rowspan=1 colspan=1>12.0</td><td rowspan=2 colspan=2>12.0</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>19.0</td></tr><tr><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>14.0</td></tr></table>

## Recall the Intuition

·Gradients generates when a variable is involved in a computation:

$$
z _ { 1 } = w _ { 1 } a + b _ { 1 }
$$

$$
z _ { 2 } = w _ { 2 } a + b _ { 2 }
$$

$$
\frac { \partial J } { \partial a } = \sum _ { j } \frac { \partial J } { \partial z _ { j } } \mathrm { ~ \cdot ~ } \frac { \partial z _ { j } } { \partial a }
$$

·Another Case:

$$
z _ { 1 } = w a _ { 1 } + b
$$

$$
z _ { 2 } = w a _ { 2 } + b
$$

$$
\begin{array} { r } { \frac { \partial J } { \partial w } = \displaystyle \sum _ { j } \frac { \partial J } { \partial z _ { j } } \cdot \frac { \partial z _ { j } } { \partial w } } \\ { \frac { \partial J } { \partial b } = \displaystyle \sum _ { j } \frac { \partial J } { \partial z _ { j } } \cdot \frac { \partial z _ { j } } { \partial b } } \end{array}
$$

## Full Operation Case - Derivatives to Parameters

·Considering One Operation of One Kernel $y _ { \mathit { I o c a l } } = W * \ x _ { \mathit { I o c a l } } + b$ (逐元素乘后相加) This generates the following gradients:

$$
\begin{array} { r } { \frac { \partial J } { \partial W } = \sum _ { I o c a l } \frac { \partial J } { \partial y _ { I o c a l } } \times \frac { \partial y _ { I o c a l } } { \partial W } = \sum _ { I o c a l } \frac { \partial J } { \partial y _ { I o c a l } } \times X _ { I o c a l } } \end{array}
$$

$$
\begin{array} { r } { \frac { \partial J } { \partial b } = \sum _ { I o c a I } \frac { \partial J } { \partial y _ { I o c a I } } \times \frac { \partial y _ { I o c a I } } { \partial \mathfrak { b } } = \sum _ { I o c a I } \frac { \partial J } { \partial y _ { I o c a I } } } \end{array}
$$

What about $\frac { \partial J } { \partial X } ?$

<!-- image-->

<!-- image-->

## Recall the Intuition

· Gradients generates when a variable is involved in a computation:

$$
z _ { 1 } = w _ { 1 } \circ + b _ { 1 }
$$

$$
z _ { 2 } = w _ { 2 } a + b _ { 2 }
$$

$$
\frac { \partial J } { \partial a } = \sum _ { j } \frac { \partial J } { \partial z _ { j } } \mathrm { ~ \cdot ~ } \frac { \partial z _ { j } } { \partial a }
$$

·Another Case:

$$
\begin{array} { r } { z _ { 1 } = w a _ { 1 } + b } \\ { z _ { 2 } = w a _ { 2 } + b } \end{array}
$$

$$
\begin{array} { r } { \frac { \partial J } { \partial w } = \displaystyle \sum _ { j } \frac { \partial J } { \partial z _ { j } } \cdot \frac { \partial z _ { j } } { \partial w } } \\ { \frac { \partial J } { \partial b } = \displaystyle \sum _ { j } \frac { \partial J } { \partial z _ { j } } \cdot \frac { \partial z _ { j } } { \partial b } } \end{array}
$$

How Many Results in the green matrix does the Purple Square contribute to?

<table><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>02</td><td rowspan=1 colspan=1>02</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>1,1</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td></tr></table>

<table><tr><td></td><td>12.012.017.0</td><td></td></tr><tr><td></td><td>10.017.019.0</td><td></td></tr><tr><td>9.0</td><td>6.0</td><td>14.0</td></tr></table>

<table><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>02</td><td rowspan=1 colspan=1>02</td><td rowspan=1 colspan=1>1。0</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td></tr></table>

<table><tr><td>12.01</td><td>12.0</td><td>17.0</td></tr><tr><td>10.0</td><td>17.0</td><td>19.0</td></tr><tr><td>9.0</td><td>6.0</td><td>14.0</td></tr></table>

<table><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0。0</td><td rowspan=1 colspan=1>01</td><td rowspan=1 colspan=1>1。2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>011</td><td rowspan=1 colspan=1>02</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td></tr></table>

<!-- image-->  
4 Results, 4 Gradients $\frac { \partial J } { \partial X _ { I o c a I } }$ to accumulate

<table><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>02</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td></tr></table>

<table><tr><td></td><td>12.012.017.0</td><td></td></tr><tr><td></td><td>10.017.01</td><td>19.0</td></tr><tr><td>9.0</td><td>6.0</td><td>14.0</td></tr></table>

<table><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>00</td><td rowspan=1 colspan=1>1,1</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>000</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td></tr></table>

<table><tr><td></td><td>12.012.017.0</td><td></td></tr><tr><td>10.0</td><td>17.0</td><td>19.0</td></tr><tr><td>9.0</td><td>6.0</td><td>14.0</td></tr></table>

<!-- image-->  
4 Results, 4 Gradients $\frac { \partial J } { \partial X _ { I o c a I } }$ to accumulate

$$
\frac { \partial J } { \partial X _ { I o c a l } }
$$

## Full Operation Case - Derivatives to Parameters

Add all $\frac { \partial J } { \partial X _ { I o c a I } }$ to their own location

We can get $\frac { \partial J } { \partial X }$

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

## What about Case in reality?

· Input Shape: $C _ { i n } \times H _ { i n } \times W _ { i n }$

· Kernel Shape: $C _ { o u t } \times C _ { i n } \times H \times W$

· Bias Shape: $\complement _ { o u t }$

· Stride S, Padding P

·Output Shape: $C _ { o u t } \times H _ { o u t } \times W _ { o u t }$

<!-- image-->

## Pooling Layer - Simplified Conv

Forward

<table><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>&gt;</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr></table>

<!-- image-->

Backward  
<!-- image-->

## Pooling Layer - Simplified Conv

Forward

<table><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr></table>

<!-- image-->

Backward

<table><tr><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td></tr><tr><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td></tr><tr><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr></table>

<!-- image-->

## Operators in PyTorch,MindSpore

·Every variable is modeled as a \`Tensor':

${ \sf W } , { \sf b } , { \sf X } , { \sf Y }$

·When a \`Tensor\` being operated it will be connected to the computation graph(计算图)

· In a computation graph, each node is a Tensor, and each edge represents an operation

· Grads are automatically computed.

<!-- image-->