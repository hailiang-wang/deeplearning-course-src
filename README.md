# 深度学习课程 项目代码

https://gitee.com/hailiang-wang/deeplearning-course-src

Get started with Deep Learning using PyTorch.

# 教材

https://gitee.com/hlcap/books.tech/tree/master/AI

Pytorch深度学习实战.pdf

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
scripts/pip_install.sh torchinfo pandas numpy matplotlib
```


## 执行代码

```
./scripts/python_exec.sh notebooks/xxx.py
e.g. 
    ./scripts/python_exec.sh practice_6_2.py
```