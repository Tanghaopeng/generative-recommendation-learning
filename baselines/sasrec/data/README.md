# SASRec 输入数据

在仓库根目录运行 `python scripts/prepare_office.py` 后，
这里会生成 `office2018.txt`，每行是已经按用户、时间排列的 `user_id item_id`。

数据文件不提交到 Git；其统计与校验记录位于根目录 `experiments/`。
从 `baselines/sasrec` 启动 `main.py`，上游脚本才能正确读取这个相对路径。
