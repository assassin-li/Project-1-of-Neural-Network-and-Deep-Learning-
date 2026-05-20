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

<table><tr><td rowspan=1 colspan=1> $w _ { 1 }$ </td><td rowspan=1 colspan=1> $\boldsymbol { \mathsf { W } } _ { 2 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 3 }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 4 }$ </td><td rowspan=1 colspan=1> ${ \mathsf W } _ { 5 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 6 }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathsf { w } _ { 7 }$ </td><td rowspan=1 colspan=1> $\mathsf { w } _ { 8 }$ </td><td rowspan=1 colspan=1>W9</td></tr></table>

l W

<!-- image-->  
( pad ) l + +1 d

$$
W _ { _ { l + 1 } } = \frac { ( W _ { _ { l } } + 2 \times p a d - \mathrm { k e r } n e l ) } { s t r i d e } + 1
$$

<!-- image-->

## 4. Backpropagation on CNN

Backpropagation on convolution layer

<table><tr><td rowspan=1 colspan=1>W1</td><td rowspan=1 colspan=1>W2</td><td rowspan=1 colspan=1>W3</td></tr><tr><td rowspan=1 colspan=1>W4</td><td rowspan=1 colspan=1>W5</td><td rowspan=1 colspan=1>W6</td></tr><tr><td rowspan=1 colspan=1>W7</td><td rowspan=1 colspan=1>W8</td><td rowspan=1 colspan=1>W9</td></tr></table>

$$
{ [ \begin{array} { l l } { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 1 } & { { | \begin{array} { l l } { 0 } \end{array} | } 1 } \end{array} ] } \\ { { | \begin{array} { l l } { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 1 } & { { | \begin{array} { l l } { 0 } \end{array} | } 2 } & { { | \begin{array} { l l } { 3 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 1 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \end{array} } } \\   | \begin{array} { l l } { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 1 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } &   | \begin{array}  \end{array} \end{array}
$$

( pad ) l + +1 d

$$
W _ { _ { l + 1 } } = \frac { ( W _ { _ { l } } + 2 \times p a d - \mathrm { k e r } n e l ) } { s t r i d e } + 1
$$

<!-- image-->

## 4. Backpropagation on CNN

Backpropagation on convolution layer

<table><tr><td rowspan=1 colspan=1> $^ 6 M$ </td><td rowspan=1 colspan=1> $^ { 8 } M$ </td><td rowspan=1 colspan=1> $\angle _ { M }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathfrak { s } _ { M }$ </td><td rowspan=1 colspan=1> ${ \mathsf { s } } _ { M }$ </td><td rowspan=1 colspan=1> ${ \mathfrak { p } } _ { M }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathfrak { E } _ { M }$ </td><td rowspan=1 colspan=1> $z _ { M }$ </td><td rowspan=1 colspan=1>M</td></tr></table>

(rotation)

$$
{ [ \begin{array} { l l } { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 1 } & { { | \begin{array} { l l } { 0 } \end{array} | } 1 } \end{array} ] } \\ { { | \begin{array} { l l } { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 1 } & { { | \begin{array} { l l } { 0 } \end{array} | } 2 } & { { | \begin{array} { l l } { 3 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 1 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \end{array} } } \\   | \begin{array} { l l } { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 1 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } & { { | \begin{array} { l l } { 0 } \end{array} | } 0 } \\ { 0 } &   | \begin{array}  \end{array} \end{array}
$$

( pad ) l + +1 d

$$
W _ { _ { l + 1 } } = \frac { ( W _ { _ { l } } + 2 \times p a d - \mathrm { k e r } n e l ) } { s t r i d e } + 1
$$

<!-- image-->

## 4. Backpropagation on CNN

Backpropagation on convolution layer

<!-- image-->  
(rotation)

wl  
<!-- image-->  
stride=1 pad=2  
81+1(+ pad)

<!-- image-->  
s

a. $\delta ^ { l + 1 } + p a d$

b. rotate l W $1 8 0 ^ { \circ }$

c. compute (convolution)l d

d. compute gradient $\frac { \partial } { \partial W } J$

e. update weights

## Reference

2. Blogs: http://www.cnblogs.com/tornadomeet/tag/Deep%20Learning/ http://blog.csdn.net/zouxy09/article/category/1387932

1. UFLDL: http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial

THANK YOU