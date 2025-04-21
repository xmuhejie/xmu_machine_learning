# 量化简单策略

用来实现简单的量化策略代码

## 安装过程

1.新建一个cond环境：conda create -n quant python=3.10即可
2.激活conda环境：conda activate quant
3.先将requirements.txt文件下载到当前目录的文件夹中即可，然后输入：pip install -r requirements.txt即可安装全部包，若是觉得安装比较慢的话可以考虑增加清华镜像源或者阿里镜像源：
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/或者进行国内镜像源的永久配置，具体步骤见requirements.txt，注意和conda install packages不同，pip设置的只适用于pip install packages

## 运行策略

在终端激活环境后输入：python 你的文件当前路径或者绝对路径，看你是否添加了环境变量，最后使用绝对路径

## 已有策略
* 海龟策略
* RSI策略
* 布林带策略
* 机器学习策略
* 动量因子策略
* 多因子策略: 动量因子+波动率因子+市值因子

## 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。
