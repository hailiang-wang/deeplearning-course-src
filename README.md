# 深度学习课程 项目代码

Get started with Deep Learning using PyTorch.

https://gitee.com/hailiang-wang/deeplearning-course-src

OR [GitHub](https://github.com/hailiang-wang/deeplearning-course-src).

# 教材

https://gitee.com/hlcap/books.tech/tree/master/AI

* 首选： Pytorch深度学习实战.pdf

# 运行代码

```
chmod +x scripts/*.sh
```

## 安装依赖

* CPU Machine

```
scripts/pip_install.sh torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
scripts/pip_install.sh torchinfo pandas numpy matplotlib
```

* GPU Machine

```
./scripts/install_deps.sh torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu
126 
scripts/pip_install.sh torchinfo pandas numpy matplotlib
```


## 执行代码

```
./scripts/python_exec.sh notebooks/xxx.py
e.g. 
    ./scripts/python_exec.sh practice_6_2.py
```