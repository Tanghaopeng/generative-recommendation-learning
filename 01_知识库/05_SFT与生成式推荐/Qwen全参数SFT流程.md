# 从商品编码到 Qwen 全参数 SFT

## 先明确训练哪一部分

```text
商品文本 → 冻结MiniLM → 384维向量 → RQ-VAE / EMA → 四段SID表（固定）
用户交互历史 → 查固定SID表 → Qwen聊天输入 → 逐token预测下一商品SID
                                              ↓ 交叉熵反向传播
                                    更新Qwen的全部参数
```

MiniLM已经训练好，负责内容表征；RQ-VAE已经学好离散码本。本阶段不运行它们的forward，更不回传梯度，只查保存的SID表。
Qwen2.5-1.5B-Instruct也是预训练并经指令训练的模型；推荐SFT用监督样本进一步更新它，使它学会“这种交互历史后应出现哪个商品”。
它的内部SID token embedding会随SFT更新；这些参数与MiniLM产出的384维商品向量是两套不同的对象。
第四段SID是消歧编号，无独立语义层含义。

## 1. 训练样本从哪里来

假设一位用户按时间购买 `A B C D E F G`。`F`为验证、`G`为测试，训练只用`A B C D E`。

| 样本 | 输入历史 | 监督下一商品 |
|---|---|---|
| 1 | A | B |
| 2 | A B | C |
| 3 | A B C | D |
| 4 | A B C D | E |

一位用户贡献多条next-item样本，每条只预测下一件。验证输入`A B C D E`，目标`F`；测试输入`A B C D E F`，目标`G`。
这里预测的是下一次观测到的商品，不是一次输出未来全部消费列表，也不宣称完成了OneRec列表生成。

为与已有SASRec/TIGER严格保持正样本位置一致，长用户序列只取训练末尾最多21件对应的20个目标位置：`start=max(0,len(train)-1-20)`，目标从`start+1`到最后，历史从这个固定start到目标前一项。
因此不是把所有长序列位置全部展开。Office Products实际为397,345个训练目标/轮、86,713个用户、25,898件商品。准备脚本会核验没有验证/测试交互混入训练。
商品目录和文本编码使用固定完整目录，与已有基线相同；这不等同于严格全局时间切分或新商品冷启动实验。

## 2. 一条样本的输入和输出

以下SID为虚构示例，不代表实际商品：

```text
A = <sid_0_12><sid_1_43><sid_2_7><sid_3_0>
B = <sid_0_12><sid_1_80><sid_2_5><sid_3_1>
C = <sid_0_92><sid_1_16><sid_2_3><sid_3_0>
```

“已看A、B，下一件C”的聊天结构为：

```text
system: You are a product recommendation assistant.
user: Predict the next product. Return only its four SID tokens.
      History (oldest to newest):
      <sid_0_12><sid_1_43><sid_2_7><sid_3_0>
      <sid_0_12><sid_1_80><sid_2_5><sid_3_1>
assistant: <sid_0_92><sid_1_16><sid_2_3><sid_3_0><|im_end|>
```

实际用官方tokenizer的`apply_chat_template(...,add_generation_prompt=True)`构造提示，包含Qwen自己的角色边界。
历史商品按时间从旧到新排列；默认最多20件，每件4个SID token，额外还有换行和指令token。训练与推理使用同一prompt函数。
长样本先从最旧的完整商品开始删，绝不把一个SID截成两段；标题→SID的标题输入最多128token，标题方向回答最多64token含EOS；仍超长的提示会报错。

## 3. tokenizer与词表扩展

原始Qwen词表并不认识我们的SID。添加4个位置命名空间，各256个普通原子token，共1024个：`<sid_0_0>`…`<sid_3_255>`。
这使一个商品恰好编码成4个token，不能让`<sid_0_12>`被原BPE切成多个字符碎片。逐条校验完整目录SID均为4token且唯一。
原Qwen tokenizer长度151665，扩展后152689；模型原embedding行数151936含预留槽，resize后为152689。
因此tokenizer增加1024个词条，模型embedding实际增加753行；已有预留位置也会成为SID token位置。新增行采用固定seed下的模型默认随机初始化，预留行保留原值，然后参与全参更新。
保存扩展后的tokenizer，恢复和推理必须使用同一份，SID编号不能重新分配。

## 4. 拼接、mask与teacher forcing

令`P`为完整prompt（含assistant起始标记），目标为`c0 c1 c2 c3 EOS`：

```text
input_ids:      [ P0 P1 ... Pn  c0   c1   c2   c3   EOS  PAD ... ]
attention_mask: [  1  1 ...  1   1    1    1    1     1    0 ... ]
labels:         [-100 ... -100  c0   c1   c2   c3   EOS -100 ... ]
```

Qwen是因果Transformer：每个位置只能看它之前和当前的位置。训练时目标token也放进输入，这叫teacher forcing；预测`c1`时可以看到真实`c0`，但看不到`c1`后面的内容。
模型内部自动用`logits[:,:-1]`对`labels[:,1:]`，所以prompt最后位置预测`c0`，`c0`位置预测`c1`，依此类推。调用方不能再手工shift一次。
`-100`表示这个label不计损失，**不表示这些历史token不参与计算**：它们仍通过自注意力影响目标，梯度仍能更新产生历史表示的Qwen参数。
EOS是回答结束符，参与监督；padding可能与EOS使用同一个ID，但padding位置单独mask为-100，不能简单按token值把所有EOS一起屏蔽。

## 5. forward究竟经过什么

`input_ids` → 查Qwen内部可训练embedding → 28层因果自注意力/RoPE/MLP等 → 输出隐状态 `[B,L,1536]` → LM head → 词表logits `[B,L,152689]` → 正确位置的交叉熵。
同一件历史商品用4个token表示，不是把384维MiniLM向量直接塞进Qwen；用户ID也不在输入里。
主任务每条只有4段SID加EOS的5个有效监督token。交叉熵使正确下一token的概率升高，整个扩展词表参与softmax；这里不抽“1正99负”的商品集合，也不是SASRec的正负BCE。
训练时不做前缀树mask，推理时才约束合法SID；因此模型先学会在原文本词表和SID词表中预测正确回答。

## 6. backward和全参数更新

`loss.backward()` → 目标位置的误差信号经LM head、自注意力和MLP传回所有Qwen层及内部token embedding → 累积64个microbatch → 梯度裁剪 → AdamW更新 → 清梯度。
microbatch2×累积64意味着单GPU一次更新通常包含128条样本；最后不足一个累积组由Trainer处理。
“全参数”指所有Qwen参数都`requires_grad=True`，不是只更新新增SID词条，更不是只训练LoRA。BF16影响计算精度，不改变哪些参数被训练。
梯度检查点在backward重算部分中间激活来省显存，会增加计算时间；不会把模型层冻结。

## 7. Q1与可选Q2

Q1只做“历史SID→下一SID”，先建立预训练模型推荐基线。
Q2从同一原始Qwen独立初始化，增加三种方向：标题→SID、SID→标题、历史SID→下一标题。标题缺失时跳过对应辅助样本；历史→下一标题只从训练目标构建。
全目录标题↔SID只用静态商品文本，不使用验证/测试行为标签；这是固定目录设置，不能据此声称冷启动效果。
四方向与上游MiniOneRec单卡实现的任务思路一致，但prompt、四段SID、数据划分和选模实现不同。Q2包含更多样本/训练计算，需另做等预算对照。

## 8. 推理和线上角度

线上拿到用户真实已知历史 → 查冻结SID映射 → 同训练prompt → Qwen逐段生成4token → 前缀树约束每一步只能走向目录中的商品 → 映射回唯一item_id → 过滤全部已知历史 → 返回最多20件候选。
生成时没有真实下一商品标签，不会把目标SID喂给模型。有限beam20是近似搜索，可能漏掉全目录概率更高的候选；不等同于给25,898件商品逐一精确打分。
当前beam累计四个SID token的原始条件log概率，约束之后不重新归一化，不用EOS概率重新排序；最终同分按item_id稳定排序。死分支保留负无穷并过滤，候选不足会计入shortfall，不补入真值。
检索输出是召回候选，生产环境通常还要做库存、资格过滤和业务重排；本项目目前只评测下一商品命中与排名，不含线上服务部署。
新商品要先编码、量化并更新SID索引；新SID组合是否能被推荐，需要后续增量更新与冷启动实验验证。

## 9. 什么才算实验完成

CPU微型Qwen通过梯度、mask、因果loss、恢复和检索检查，只说明代码通路可用；真实tokenizer检查也不代表推荐效果提升。
GPUpilot检查峰值显存、速度、loss是否有限和合法商品输出。正式训练按验证NDCG@10选模，候选检查点再做全量验证，冻结后一次测试；记录所有配置、hash、耗时、更新次数、HR/Recall/NDCG和shortfall。
已有SASRec、Adam-TIGER、EMA-TIGER对照不能删掉。Qwen未训练时结果填“未运行”，不能填0或引用上游数字充当本地结果。

参考：[上游固定SFT源码](https://github.com/wbn11/minionerec-single-gpu/blob/bccd7ef70291c5b4b0c30b6222bc491039b4cd18/src/minionerec/training/sft_training.py)、[Qwen官方模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)。
