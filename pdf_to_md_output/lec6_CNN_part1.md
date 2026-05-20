# Convolutional Neural Network & Backpropagation Algorithm

Xuelin Qian

## Content

1.Neural Network

2.Backpropagation Algorithm

3.Convolutional Neural Network

4.Backpropagation on CNN

## Content

1.Neural Network

2.Backpropagation Algorithm

3.Convolutional Neural Network

4.Backpropagation on CNN

## 1. Neural Network

“neuron”

<!-- image-->

$$
h _ { W , b } ( x ) = f ( W ^ { T } x ) = f ( \Sigma _ { i = 1 } ^ { 3 } W _ { i } x _ { i } + b )
$$

where $f : \Re \mapsto \Re$ is called the activation function

## 1. Neural Network

neural network: consist of many simple “neurons”

<!-- image-->

## 1. Neural Network

## Forward Propagation

$n _ { l }$ : the number of layers

<!-- image-->

$L _ { \iota }$ : the layer l

$W _ { i j } ^ { ( l ) }$ : the weight associated with the connection between unit $j$ in layer $l { . }$ and unit $i$ in layer $l + 1$

$b _ { i } ^ { ( l ) }$ : the bias associated with unit $i$ in layer $l + 1$

$a _ { i } ^ { ( l ) }$ : the acctivation unit in layer i l

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

## 1. Neural Network

## Forward Propagation

<!-- image-->

$n _ { l }$ : the number of layers

$L _ { \iota }$ : the layer l

$\it { W } _ { i j } ^ { ( l ) }$ : the weight associated with the connection between unit $j$ in layer ,l and unit $i$ in layer $l + 1$

$b _ { i } ^ { ( l ) }$ : the bias associated with unit in layer i $l + 1$

$a _ { i } ^ { ( l ) }$ : the acctivation unit in layer i l

$\boldsymbol { z } _ { i } ^ { ( l ) }$ : the total weighted sum of inputs to unit in layer i l

$$
a _ { 1 } ^ { ( 2 ) } = f ( W _ { 1 1 } ^ { ( 1 ) } x _ { 1 } + W _ { 1 2 } ^ { ( 1 ) } x _ { 2 } + W _ { 1 3 } ^ { ( 1 ) } x _ { 3 } + b _ { 1 } ^ { ( 1 ) } )\tag{e.g.}
$$

$$
a _ { 1 } ^ { ( 2 ) } = f \bigl ( z _ { 1 } ^ { ( 2 ) } \bigr ) \qquad z _ { 1 } ^ { ( 2 ) } = \sum _ { j = 1 } ^ { 3 } W _ { 1 j } ^ { ( 1 ) } x _ { j } + b _ { 1 } ^ { ( 1 ) }
$$

## 1. Neural Network

## Forward Propagation

$n _ { l }$ : the number of layers

<!-- image-->

$L _ { \iota }$ : the layer l

$\it { W } _ { i j } ^ { ( l ) }$ : the weight associated with the connection between unit $j$ in layer , l and unit $i$ in layer $l + 1$

$b _ { i } ^ { ( l ) }$ : the bias associated with unit $i$ in layer $l + 1$

$a _ { i } ^ { ( l ) }$ : the acctivation unit in layer i l

$\boldsymbol { z } _ { i } ^ { ( l ) }$ : the total weighted sum of inputs to unit in layer i l

$$
z ^ { ( 2 ) } = W ^ { ( 1 ) } x + b ^ { ( 1 ) }
$$

$$
a ^ { ( 2 ) } = f ( z ^ { ( 2 ) } )
$$

$$
z ^ { ( 3 ) } = W ^ { ( 2 ) } a ^ { ( 2 ) } + b ^ { ( 2 ) }
$$

$$
h _ { W , b } ( x ) = a ^ { ( 3 ) } = f ( z ^ { ( 3 ) } )
$$

<!-- image-->

$$
\boldsymbol { z } ^ { ( l + 1 ) } = \boldsymbol { W } ^ { ( l ) } \boldsymbol { a } ^ { ( l ) } + \boldsymbol { b } ^ { ( l ) }
$$

$$
a ^ { ( l + 1 ) } = f ( z ^ { ( l + 1 ) } )
$$

## Content

1.Neural Network

2.Backpropagation Algorithm

3.Convolutional Neural Network

4.Backpropagation on CNN

## 2. Backpropagation Algorithm

“Loss function”

<!-- image-->

$$
J ( W , b ; x , y ) = \frac { 1 } { 2 } \left\| h _ { W , b } ( x ) - y \right\| ^ { 2 } .
$$

$$
\begin{array} { c l c r } { { J ( W , b ) = \displaystyle \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } J ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } ) \right] + \frac { \lambda } { 2 } \sum _ { l = 1 } ^ { n _ { l } - 1 } \sum _ { i = 1 } ^ { s _ { l } } \sum _ { j = 1 } ^ { s _ { l + 1 } } \left( W _ { j i } ^ { ( l ) } \right) ^ { 2 } } } \\ { { = \displaystyle \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( \frac { 1 } { 2 } \left\| h _ { W , b } ( x ^ { ( i ) } ) - y ^ { ( i ) } \right\| ^ { 2 } \right) \right] + \frac { \lambda } { 2 } \sum _ { l = 1 } ^ { n _ { l } - 1 } \sum _ { i = 1 } ^ { s _ { l } } \sum _ { j = 1 } ^ { s _ { l + 1 } } \left( W _ { j i } ^ { ( l ) } \right) ^ { 2 } } } \end{array}
$$

## 2. Backpropagation Algorithm

“Gradient Descent”

<!-- image-->

$$
\begin{array} { c } { { \displaystyle J ( W , b ) = \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } J ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } ) \right] + \displaystyle \frac { \lambda } { 2 } \sum _ { l = 1 } ^ { n - 1 } \sum _ { i = 1 } ^ { s _ { l } } \left( W _ { j i } ^ { ( l ) } \right) ^ { 2 } } } \\ { { = \displaystyle \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( \frac { 1 } { 2 } \left\| h _ { W , b } ( x ^ { ( i ) } ) - y ^ { ( i ) } \right\| ^ { 2 } \right) \right] + \displaystyle \frac { \lambda } { 2 } \sum _ { l = 1 } ^ { n - 1 } \sum _ { i = 1 } ^ { s _ { l } } \sum _ { j = 1 } ^ { s _ { l + 1 } } \left( W _ { j i } ^ { ( l ) } \right) ^ { 2 } } } \\ { { W _ { i j } ^ { ( l ) } = W _ { i j } ^ { ( l ) } - \displaystyle \frac { \sqrt { \frac { \partial } { \partial W _ { i j } ^ { ( l ) } } J ( W , b ) } } { \displaystyle \left\{ \frac { \partial W _ { i j } ^ { ( l ) } } { \partial W _ { i j } ^ { ( l ) } } \right\} } } } \\ { { b _ { i } ^ { ( l ) } = b _ { i } ^ { ( l ) } - \alpha \displaystyle \frac { \partial } { \partial b _ { i } ^ { ( l ) } } J ( W , b ) } } \end{array}
$$

## 2. Backpropagation Algorithm

“Gradient Descent”

<!-- image-->

$$
\begin{array} { l } { { J ( W , b ) = \displaystyle \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } J ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } ) \right] + \displaystyle \frac { \lambda } { 2 } \sum _ { l = 1 } ^ { n - 1 } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { s _ { i + 1 } } \left( W _ { j i } ^ { ( l ) } \right) ^ { 2 } } } \\ { { \displaystyle \qquad = \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( \frac { 1 } { 2 } \left\| h _ { W , b } ( x ^ { ( i ) } ) - y ^ { ( i ) } \right\| ^ { 2 } \right) \right] + \displaystyle \frac { \lambda } { 2 } \sum _ { l = 1 } ^ { n - 1 } \sum _ { i = 1 } ^ { s _ { i } } \sum _ { j = 1 } ^ { s _ { i + 1 } } \left( W _ { j i } ^ { ( l ) } \right) ^ { 2 } } } \\ { { \displaystyle \frac { \partial } { \partial W _ { i j } ^ { ( l ) } } J ( W , b ) = \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { \partial } { \partial W _ { i j } ^ { ( l ) } } J ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } ) \right] + \lambda W _ { i j } ^ { ( l ) } } } \\ { { \displaystyle \qquad \frac { \partial } { \partial b _ { i } ^ { ( l ) } } J ( W , b ) = \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { \partial } { \partial b _ { i } ^ { ( l ) } } J ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } ) } } \end{array}
$$

## 2. Backpropagation Algorithm

“Derivative chain rule”

<!-- image-->

$$
\begin{array} { r l } & { \frac { \partial } { \partial W _ { i j } ^ { ( l ) } } J ( W , b ) = \left[ \frac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } \displaystyle \sum _ { j = W _ { i j } ^ { ( l ) } } ^ { \frac { \partial } { \partial } } J ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } ) \right] \left[ \right] + \lambda W _ { i j } ^ { ( l ) } } \\ & { \frac { \partial } { \partial b _ { i } ^ { ( l ) } } J ( W , b ) = \frac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } \frac { \partial } { \partial b _ { i } ^ { ( l ) } } J ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } ) } \end{array}
$$

$$
z _ { i } ^ { ( l + 1 ) } = \sum _ { j = 1 } ^ { s _ { l } } W _ { i j } ^ { ( l ) } a _ { j } ^ { ( l ) } + b _ { i } ^ { ( l ) } a _ { i } ^ { \left( l + 1 \right) } = f { \left( z _ { i } ^ { ( l + 1 ) } \right) }\tag{Page 7}
$$

## 2. Backpropagation Algorithm

“error term” ，

<!-- image-->

$$
\delta _ { i } ^ { ( l + 1 ) } = \frac { \hat { O } } { \hat { O } z _ { i } ^ { ( l + 1 ) } } J \big ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } \big )
$$

which measures how much that node was "responsible" for any errors in output

## 2. Backpropagation Algorithm

“error term”

$$
\delta _ { i } ^ { ( l + 1 ) } = \frac { \hat { \partial } } { \hat { \partial } z _ { i } ^ { ( l + 1 ) } } J \big ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } \big )
$$

a) For each output unit in layer i $n _ { l }$

<!-- image-->

$$
\begin{array} { c } { \displaystyle \delta _ { i } ^ { ( n _ { l } ) } = \frac { \partial } { \partial z _ { i } ^ { n _ { l } } } J ( W , b ; x , y ) = \frac { \partial } { \partial z _ { i } ^ { n _ { l } } } \frac { 1 } { 2 } \left\| y - h _ { W , b } ( x ) \right\| ^ { 2 } } \\ { \displaystyle = \frac { \partial } { \partial z _ { i } ^ { n _ { l } } } \frac { 1 } { 2 } \sum _ { j = 1 } ^ { S _ { n _ { l } } } ( y _ { j } - a _ { j } ^ { ( n _ { l } ) } ) ^ { 2 } = \frac { \partial } { \partial z _ { i } ^ { n _ { l } } } \frac { 1 } { 2 } \sum _ { j = 1 } ^ { S _ { n _ { l } } } ( y _ { j } - f ( z _ { j } ^ { ( n _ { l } ) } ) ) ^ { 2 } } \\ { \displaystyle = - ( y _ { i } - f ( z _ { i } ^ { ( n _ { l } ) } ) ) \cdot f ^ { \prime } ( z _ { i } ^ { ( n _ { l } ) } ) = - ( y _ { i } - a _ { i } ^ { ( n _ { l } ) } ) \cdot f ^ { \prime } ( z _ { i } ^ { ( n _ { l } ) } ) } \end{array}
$$

(Page 7)

## 2. Backpropagation Algorithm

“error term”

b) For $l = n _ { l } - 1 , n _ { l } - 2 , n _ { l } - 3 , \cdots , 2$

<!-- image-->

$$
\begin{array} { r l } { \delta _ { i } ^ { ( n - 1 ) } - \displaystyle \frac { \partial } { \partial x _ { i } ^ { n - 1 } } J ( W , t , x , y ) - \frac { \partial } { \partial x _ { i } ^ { n - 1 } } \frac { 1 } { 2 } \| y - h _ { W , ( i ) } ( x ) \| ^ { 2 } - \frac { \partial } { \partial x _ { i } ^ { n - 1 } } \frac { 1 } { 2 } \sum _ { \tilde { \rho } = \frac { - i } { 1 } } \frac { \partial } { \partial y _ { i } ^ { n } } ( y - a _ { i } ^ { [ n ] } ) ^ { 2 } } & { } \\ { - \displaystyle \frac { 1 } { 2 } \sum _ { s = 1 } ^ { N } \frac { \partial } { \partial x _ { i } ^ { n - 1 } } ( y _ { i } - a _ { j } ^ { [ n ] } ) ^ { 2 } - \frac { 1 } { 2 } \sum _ { j = 1 } ^ { N } \frac { \partial } { \partial x _ { i } ^ { n - 1 } } ( y _ { i } - I ( z _ { j } ^ { [ n ] } ) ) ^ { 2 } } & { } \\ { - \displaystyle \sum _ { s = 1 } ^ { N } - ( y _ { i } - f ( z _ { i } ^ { [ n ] } ) ) \cdot \frac { \partial } { \partial x _ { i } ^ { [ n ] } } ( y _ { i } ^ { [ n ] } ) - \sum _ { s = 1 } ^ { N } - ( y _ { j } - f ( z _ { i } ^ { [ n ] } ) ) \cdot f ( z _ { j } ^ { [ n ] } ) ) \cdot \frac { \partial ^ { 2 } ( n ) } { \partial z _ { i } ^ { [ n - 1 ] } } } & { } \\ { - \displaystyle \sum _ { s = 1 } ^ { N } \delta _ { i } ^ { ( n ) } - \frac { \partial ^ { 2 } ( n ) } { \partial x _ { i } ^ { [ n ] } } - \sum _ { j = 1 } ^ { N } \left( \delta _ { j } ^ { [ n ] } \cdot \frac { \partial } { \partial x _ { i } ^ { [ n ] } } - \sum _ { k = 1 } ^ { N } - ( y _ { i } ^ { [ n ] } ) \cdot W _ { k } ^ { [ n ] } \right) } \end{array}
$$

## 2. Backpropagation Algorithm

“error term”

b) For $l = n _ { l } - 1 , n _ { l } - 2 , n _ { l } - 3 , \cdots , 2$

<!-- image-->

$$
\delta _ { i } ^ { ( n _ { l } - 1 ) } = \left( \sum _ { j = 1 } ^ { S _ { n _ { l } } } W _ { j i } ^ { n _ { l } - 1 } \delta _ { j } ^ { ( n _ { l } ) } \right) f ^ { \prime } ( z _ { i } ^ { n _ { l } - 1 } )
$$

$$
\delta _ { i } ^ { ( l ) } = \left( \sum _ { j = 1 } ^ { \cdot _ { l + 1 } } W _ { j i } ^ { ( l ) } \delta _ { j } ^ { ( l + 1 ) } \right) f ^ { \prime } ( z _ { i } ^ { ( l ) } )
$$

## 2. Backpropagation Algorithm

“Gradient Descent”

$$
\boldsymbol { W } _ { i j } ^ { ( l ) } = \boldsymbol { W } _ { i j } ^ { ( l ) } - \alpha \frac { \partial } { \partial \boldsymbol { W } _ { i j } ^ { ( l ) } } \boldsymbol { J } ( \boldsymbol { W } , \boldsymbol { b } )
$$

$$
b _ { i } ^ { ( l ) } = b _ { i } ^ { ( l ) } - \alpha \frac { \partial } { \partial b _ { i } ^ { ( l ) } } J ( W , b )
$$

$$
\frac { \partial } { \partial W _ { i j } ^ { ( l ) } } J ( W , b ) = \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left| \overline { { { \frac { \partial } { \partial W _ { i j } ^ { ( l ) } } J ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } ) } } } \right| \right] + \lambda W _ { i j } ^ { ( l ) }
$$

“Gradient Descent”

$$
\frac { \partial } { \partial b _ { i } ^ { ( l ) } } J ( W , b ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { \partial } { \partial b _ { i } ^ { ( l ) } } J ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } )
$$

“Derivative chain rule”

$$
\frac { \hat { \sigma } } { \hat { \sigma } W _ { i j } ^ { ( l ) } } J \big ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } \big ) = \frac { \hat { \sigma } } { \hat { \sigma } z _ { i } ^ { ( l + 1 ) } } J \big ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } \big ) \times \frac { \hat { \sigma } z _ { i } ^ { ( l + 1 ) } } { \hat { \sigma } W _ { i j } ^ { ( l ) } }
$$

“error term” ，

$$
\stackrel { \longleftrightarrow } { \underbrace { \dot { S } _ { i } ^ { ( l + 1 ) } } } = \frac { \hat { Q } } { \hat { Q } z _ { i } ^ { ( l + 1 ) } } J \big ( W , b ; x ^ { ( i ) } , y ^ { ( i ) } \big ) \qquad \stackrel { \longleftrightarrow } { \underbrace { \big ( z _ { i } ^ { ( l + 1 ) } = \sum _ { j = 1 } ^ { s _ { l } } W _ { i j } ^ { ( l ) } a _ { j } ^ { ( l ) } + b _ { i } ^ { ( l ) } \big ) } }
$$

## 2. Backpropagation Algorithm

$$
\begin{array} { c } { { \displaystyle { \frac { \partial } { \partial W _ { i j } ^ { ( l ) } } } J ( W , b ; x , y ) = a _ { j } ^ { ( l ) } \delta _ { i } ^ { ( l + 1 ) } } } \\ { { \displaystyle { \frac { \partial } { \partial b _ { i } ^ { ( l ) } } } J ( W , b ; x , y ) = \delta _ { i } ^ { ( l + 1 ) } . } } \end{array}
$$

“error term” ）

$$
\begin{array} { l } { { \delta _ { i } ^ { ( n _ { l } ) } = \displaystyle \frac { \partial } { \partial z _ { i } ^ { ( n _ { l } ) } } ~ \frac { 1 } { 2 } \left\| y - h _ { W , b } ( x ) \right\| ^ { 2 } = - ( y _ { i } - a _ { i } ^ { ( n _ { l } ) } ) \cdot f ^ { \prime } ( z _ { i } ^ { ( n _ { l } ) } ) } } \\ { { \delta _ { i } ^ { ( l ) } = \displaystyle \left( \sum _ { j = 1 } ^ { s _ { l + 1 } } W _ { j i } ^ { ( l ) } \delta _ { j } ^ { ( l + 1 ) } \right) f ^ { \prime } ( z _ { i } ^ { ( l ) } ) } } \end{array}
$$

## 2. Backpropagation Algorithm

$$
\begin{array} { c } { { \displaystyle { \frac { \partial } { \partial W _ { i j } ^ { ( l ) } } } J ( W , b ; x , y ) = a _ { j } ^ { ( l ) } \delta _ { i } ^ { ( l + 1 ) } } } \\ { { \displaystyle { \frac { \partial } { \partial b _ { i } ^ { ( l ) } } } J ( W , b ; x , y ) = \delta _ { i } ^ { ( l + 1 ) } . } } \end{array}
$$

<!-- image-->

$$
\delta _ { i } ^ { ( n _ { l } ) } = { \frac { \partial } { \partial z _ { i } ^ { ( n _ { l } ) } } } ~ { \frac { 1 } { 2 } } \left\| y - h _ { W , b } ( x ) \right\| ^ { 2 } = - ( y _ { i } - a _ { i } ^ { ( n _ { l } ) } ) \cdot f ^ { \prime } ( z _ { i } ^ { ( n _ { l } ) } )
$$

“error term”

$$
\delta _ { i } ^ { ( l ) } = \left( \sum _ { j = 1 } ^ { s _ { l + 1 } } W _ { j i } ^ { ( l ) } \delta _ { j } ^ { ( l + 1 ) } \right) f ^ { \prime } ( z _ { i } ^ { ( l ) } )
$$