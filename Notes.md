
# 关于BinaryNet的一些笔记

## 算法实现细节

### 基于移位操作实现的近似乘法

在BinaryNet实现过程中，为了节省乘法器资源消耗，节约运行时间，Batch Normaization算法和ADAM算法使用了移位操作近似乘法。

对于乘法运算$z=x\times y$，可以将$y$展开为：
$$y=a_n2^n+a_{n-1}2^{n-1}+a_{n-2}2^{n-2}+\cdots$$
因此原式$z=x\times y$等价于：
$$y=x\cdot a_n2^n+x\cdot a_{n-1}2^{n-1}+x\cdot a_{n-2}2^{n-2}+\cdots$$
将$x$与$a_i$交换位置，并且利用位操作中：${x\cdot 2^{i}=x<<i}$ ($i\in Z$)
可以得到
$$y=a_n(x<<n)+a_{n-1}(x<<(n-1))+a_{n-2}(x<<(n-2))+\cdots$$
若有$a_i=\pm 1$，则在上式仅由移位运算与加法构成，不包含乘法运算。

在算法中，为了满足$a_i=\pm 1$这个条件，作者引入了一个近似函数：
$$x\approx AP2(x)=sign(x)2^{round(log_2x)}$$
显然：$a_n=sing(x)=\pm 1$，因此：
$$z=x\times y\approx x\times AP2(y)=sign(y)(x<<(round(log_2x)))$$