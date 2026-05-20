## 2. Backpropagation Algorithm

$$
\begin{array} { c } { { \displaystyle { \frac { \partial } { \partial W _ { i j } ^ { ( l ) } } J ( W , b ; x , y ) = a _ { j } ^ { ( l ) } \delta _ { i } ^ { ( l + 1 ) } } } } \\ { { \displaystyle { \frac { \partial } { \partial b _ { i } ^ { ( l ) } } J ( W , b ; x , y ) = \delta _ { i } ^ { ( l + 1 ) } } . } } \end{array}
$$

<!-- image-->

$$
\delta _ { i } ^ { ( n _ { l } ) } = { \frac { \partial } { \partial z _ { i } ^ { ( n _ { l } ) } } } ~ { \frac { 1 } { 2 } } \left\| y - h _ { W , b } ( x ) \right\| ^ { 2 } = - ( y _ { i } - a _ { i } ^ { ( n _ { l } ) } ) \cdot f ^ { \prime } ( z _ { i } ^ { ( n _ { l } ) } )
$$

“error term”

$$
\delta _ { i } ^ { ( l ) } = \left( \sum _ { j = 1 } ^ { s _ { l + 1 } } W _ { j i } ^ { ( l ) } \delta _ { j } ^ { ( l + 1 ) } \right) f ^ { \prime } ( z _ { i } ^ { ( l ) } )
$$

结论：一个点（神经元）的残差=前向的时候下一层中使用过该点的神经单元的残差按权重求和

## Content

1.Neural Network

2.Backpropagation Algorithm

3.Convolutional Neural Network

4.Backpropagation on CNN

## 3. Convolutional Neural Network

<!-- image-->

<!-- image-->

## 3. Convolutional Neural Network

<!-- image-->

<!-- image-->

## 3. Convolutional Neural Network

convolution

<!-- image-->

How to calculate the number of hidden units? 1,000,000 hidden units?

## 3. Convolutional Neural Network

## convolution layer

stride, pad, kernel(size), number(output)

$$
H _ { l + 1 } = \frac { ( H _ { l } + 2 \times p a d \_ h \_ h \_ h ) } { s t r i d e \_ h } + 1 \qquad W _ { l + 1 } = \frac { ( W _ { l } + 2 \times p a d \_ w \_ \_ w l \_ w ) } { s t r i d e \_ w } + 1
$$

$$
\begin{array} { r }  \frac  | \begin{array} { l } { \mathbf { 1 } _ { \times = } } \end{array} | \mathbf { 1 } _ { \times = } | \begin{array} { l } { \mathbf { 0 } } \\ { \frac { 0 } { \times = } } \end{array} | \begin{array} { l } { 1 _ { \times = } } \end{array} 1 | \begin{array} { l } { 0 } \\ { 1 _ { \times = } } \end{array} | \begin{array} { l } { 0 } \\ { 0 } \\ { \frac { 0 } { \times = } } \end{array} | \begin{array} { l } { 0 _ { \times = } } \end{array} \end{array}
$$

<table><tr><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr></table>

ker nel

$$
H _ { l } = W _ { l } = 5
$$

$$
p a d = 0
$$

Image

Convolved Feature

$$
s t r i d e = 1
$$

## 3. Convolutional Neural Network

pooling layer

Max Pooling & Avg Pooling

<!-- image-->  
Convolved feature

<!-- image-->  
Pooled feature

## 3. Convolutional Neural Network

fully connected layer

<!-- image-->

## activation function

<!-- image-->  
ReLU

<!-- image-->  
Sigmoid

<!-- image-->

## 3. Convolutional Neural Network

<!-- image-->

## 3. Convolutional Neural Network

convolution layer

<!-- image-->  
wl

<table><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr></table>

1

<!-- image-->  
1+1

a. compute the input size

b. setup stride, pad, kernel(size), number(output),

c. compute the size of filter and output

d. \*initialize weights (the first iteration)

e. convolution

## Content

1.Neural Network

2.Backpropagation Algorithm

3.Convolutional Neural Network

4.Backpropagation on CNN

## 4. Backpropagation on CNN

Ø Backpropagation on pooling layer

Ø Backpropagation on convolution layer

## 4. Backpropagation on CNN

$$
\boxed { \begin{array} { r l } & { \boxed { \frac { \partial } { \partial W _ { i j } ^ { ( l ) } } J ( W , b ; x , y ) = a _ { j } ^ { ( l ) } \delta _ { i } ^ { ( l + 1 ) } } } \\ & { \boxed { \frac { \partial } { \partial b _ { i \dots } ^ { ( l ) } } J ( W , b ; x , y ) = \delta _ { i } ^ { ( l + 1 ) } . } } \end{array} }
$$

<!-- image-->

$$
\delta _ { i } ^ { ( n _ { l } ) } = { \frac { \partial } { \partial z _ { i } ^ { ( n _ { l } ) } } } ~ { \frac { 1 } { 2 } } \left\| y - h _ { W , b } ( x ) \right\| ^ { 2 } = - ( y _ { i } - a _ { i } ^ { ( n _ { l } ) } ) \cdot f ^ { \prime } ( z _ { i } ^ { ( n _ { l } ) } )
$$

“error term”

$$
\delta _ { i } ^ { ( l ) } = \left( \sum _ { j = 1 } ^ { s _ { l + 1 } } W _ { j i } ^ { ( l ) } \delta _ { j } ^ { ( l + 1 ) } \right) f ^ { \prime } ( z _ { i } ^ { ( l ) } )
$$

结论：一个点（神经元）的残差=前向的时候下一层中使用过该点的神经单元的残差按权重求和

## 4. Backpropagation on CNN

Backpropagation on pooling layer

Forward

<table><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr></table>

Forward

<table><tr><td rowspan=1 colspan=1>00</td><td rowspan=1 colspan=1>00</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>0。</td><td rowspan=1 colspan=1>0。</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr></table>

$$
\longleftarrow { \frac { \delta ^ { ( l + 1 ) } \to \delta ^ { ( l ) } } { 3 \mid 4 } } 
$$

## 4. Backpropagation on CNN

Backpropagation on pooling layer

Forward

<table><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr></table>

<table><tr><td>7</td><td>11</td></tr><tr><td>11</td><td>5</td></tr></table>

Backward

<table><tr><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td></tr><tr><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td></tr><tr><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr></table>

<!-- image-->

## 4. Backpropagation on CNN

Backpropagation on convolution layer

<table><tr><td rowspan=1 colspan=1> $w _ { 1 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 2 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 3 }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 4 }$ </td><td rowspan=1 colspan=1> $\boldsymbol { \mathsf { W } } _ { 5 }$ </td><td rowspan=1 colspan=1>W6</td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 7 }$ </td><td rowspan=1 colspan=1> $w _ { 8 }$ </td><td rowspan=1 colspan=1>W9</td></tr></table>

<table><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>→</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr></table>

<!-- image-->  
Image

<!-- image-->  
Convolved Feature

<!-- image-->  
l +1

## 4. Backpropagation on CNN

Backpropagation on convolution layer

<table><tr><td rowspan=1 colspan=1>W1</td><td rowspan=1 colspan=1> $\mathsf { w } _ { 2 }$ </td><td rowspan=1 colspan=1>W3</td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 4 }$ </td><td rowspan=1 colspan=1> ${ \mathsf W } _ { 5 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 6 }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 7 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 8 }$ </td><td rowspan=1 colspan=1>W9</td></tr></table>

$$
{ \left[ \begin{array} { l l } { 0 } & { { \left| \ 0 \right| } } & { 0 } \\ { 0 } & { { \left| \ 0 \right| } } & { 0 } \\ { 0 } & { { \left| \ 0 \right| } } & { 0 } \\ { { \left| \ 0 \right| } } & { 0 } & { 1 } \\ { 0 } & { { \left| \ 0 \right| } } & { { \left| \ 2 \right| } } \\ { 0 } & { { \left| \ 0 \right| } } & { 4 } \\ { 0 } & { { \left| \ 0 \right| } } & { 0 } \end{array} \right] } \ { \left[ \begin{array} { l l } { 0 } & { { \left| \ 0 \right| } } & { 0 } \\ { 0 } & { 0 } \\ { 0 } & { 0 } \end{array} \right] } \ { \left[ \begin{array} { l } { 0 } \\ { 0 } \\ { 0 } \end{array} \right] }
$$

( pad ) l + +1 d

$$
W _ { _ { l + 1 } } = \frac { ( W _ { \scriptscriptstyle { l } } + 2 \times p a d - \mathrm { k e r } n e l ) } { s t r i d e } + 1
$$

<!-- image-->

## 4. Backpropagation on CNN

Backpropagation on convolution layer

<table><tr><td rowspan=1 colspan=1> $w _ { 1 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 2 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 3 }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 4 }$ </td><td rowspan=1 colspan=1> ${ \mathsf W } _ { 5 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 6 }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 7 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 8 }$ </td><td rowspan=1 colspan=1> $w _ { 9 }$ </td></tr></table>

$$
W ^ { l }
$$

<!-- image-->

$$
\delta ^ { l + 1 } \left( + p a d \right)
$$

$$
W _ { _ { l + 1 } } = \frac { ( W _ { _ { l } } + 2 \times p a d - \mathrm { k e r } n e l ) } { s t r i d e } + 1
$$

<!-- image-->

$$
\delta ^ { l }
$$

结论：一个点（神经元）的残差=前向的时候下一层中使用过该点的神经单元的残差按权重求和

## 4. Backpropagation on CNN

Backpropagation on convolution layer

<table><tr><td rowspan=1 colspan=1> $w _ { 1 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 2 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 3 }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 4 }$ </td><td rowspan=1 colspan=1> ${ \mathsf W } _ { 5 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 6 }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 7 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 8 }$ </td><td rowspan=1 colspan=1> $w _ { 9 }$ </td></tr></table>

l W

<!-- image-->  
( pad ) l + +1 d

$$
W _ { _ { l + 1 } } = \frac { ( W _ { _ { l } } + 2 \times p a d - \mathrm { k e r } n e l ) } { s t r i d e } + 1
$$

<!-- image-->

## 4. Backpropagation on CNN

Backpropagation on convolution layer

<table><tr><td rowspan=1 colspan=1> $w _ { 1 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 2 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 3 }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 4 }$ </td><td rowspan=1 colspan=1> ${ \mathsf W } _ { 5 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 6 }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 7 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 8 }$ </td><td rowspan=1 colspan=1> $w _ { 9 }$ </td></tr></table>

l W

<!-- image-->  
( pad ) l + +1 d

$$
W _ { _ { l + 1 } } = \frac { ( W _ { _ { l } } + 2 \times p a d - \mathrm { k e r } n e l ) } { s t r i d e } + 1
$$

<!-- image-->