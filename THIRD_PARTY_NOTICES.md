# 第三方代码与数据来源

## SASRec PyTorch 实现

- 上游：[pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)
- 作者：Zan Huang；论文作者：Wang-Cheng Kang、Julian McAuley。
- 固定版本：`b253407b3c5d6ec3201a39767a572f44fd5ef6d0`。
- 本仓库位置：`baselines/sasrec/`。
- 代码来源：上游 `python/main.py`、`python/model.py`、`python/utils.py`。
- 这三个源码文件按原样复制，没有声称为本项目原创。
- 已保留上游 LICENSE、README、CITATION.cff、Result_Norm.md。
- 许可证：Apache-2.0，见 [保留的许可证](baselines/sasrec/LICENSE)。
- 作者原始实现：[kang205/SASRec](https://github.com/kang205/SASRec)，本仓库只链接参考。

## Amazon Reviews 2018

- 官方页面：<https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/>
- 数据作者：Jianmo Ni、Jiacheng Li、Julian McAuley。
- 引用：Justifying recommendations using distantly-labeled reviews and fine-grained aspects，EMNLP 2019。
- 原始交互和元数据由下载脚本从官方地址获取，不作为本仓库源码发布。
- 数据使用应遵循原始来源的说明；本仓库的代码许可证不覆盖第三方数据。
- 下载地址、字节数、SHA-256 和读取记录数见 [sources.json](sources.json)。

## 本项目新增部分

数据下载、清洗、核验脚本，学习路线，代码讲解和实验记录由本项目整理维护。
仓库根目录 LICENSE 采用 Apache-2.0；第三方内容保留各自原有权利和署名。
