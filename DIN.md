# DIN

#### 📌DIN论文需要掌握的知识

- 背景、架构、PReLU、Dice、Mini-Batch-aware-Regulazation
- 论文地址: https://arxiv.org/pdf/1706.06978

#### 📌DIN代码需要掌握的知识

可以通过`from deepctr.models import DIN`来使用`DIN`模型，此时是使用`TensorFlow`框架构建的

```python
def DIN(dnn_feature_columns, history_feature_list, dnn_use_bn=False, dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice", att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024, task='binary'):
```

- dnn_feature_columns: 特征列， 包含数据所有特征的列表
- history_feature_list: 用户历史行为列， 反应用户历史行为的特征的列表
- dnn_use_bn: 是否使用BatchNormalization
- dnn_hidden_units: 全连接层网络的层数和每一层神经元的个数， 一个列表或者元组
- dnn_activation_relu: 全连接网络的激活单元类型
- att_hidden_size: 注意力层的全连接网络的层数和每一层神经元的个数
- att_activation: 注意力层的激活单元类型
- att_weight_normalization: 是否归一化注意力得分
- l2_reg_dnn: 全连接网络的正则化系数
- l2_reg_embedding: embedding向量的正则化稀疏
- dnn_dropout: 全连接网络的神经元的失活概率
- task: 任务， 可以是分类， 也可是是回归

`DeepCTR-torch`支持使用`Pytorch`版本的`DIN`

对于这行代码：`from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names`

1. 首先，我们要处理数据集， 得到数据， 由于我们是基于用户过去的行为去预测用户是否点击当前文章， 所以我们需要把数据的特征列划分成数值型特征， 离散型特征和历史行为特征列三部分， 对于每一部分， DIN模型的处理会有不同
   1. 对于离散型特征， 在我们的数据集中就是那些类别型的特征， 比如user_id这种， 这种类别型特征， 我们首先要经过embedding处理得到每个特征的低维稠密型表示， 既然要经过embedding， 那么我们就需要为每一列的类别特征的取值建立一个字典，并指明embedding维度， 所以在使用deepctr的DIN模型准备数据的时候， 我们需要通过SparseFeat函数指明这些类别型特征, 这个函数的传入参数就是列名， 列的唯一取值(建立字典用)和embedding维度。
   2. 对于用户历史行为特征列， 比如文章id， 文章的类别等这种， 同样的我们需要先经过embedding处理， 只不过和上面不一样的地方是，对于这种特征， 我们在得到每个特征的embedding表示之后， 还需要通过一个Attention_layer计算用户的历史行为和当前候选文章的相关性以此得到当前用户的embedding向量， 这个向量就可以基于当前的候选文章与用户过去点击过得历史文章的相似性的程度来反应用户的兴趣， 并且随着用户的不同的历史点击来变化，去动态的模拟用户兴趣的变化过程。这类特征对于每个用户都是一个历史行为序列， 对于每个用户， 历史行为序列长度会不一样， 可能有的用户点击的历史文章多，有的点击的历史文章少， 所以我们还需要把这个长度统一起来， 在为DIN模型准备数据的时候， 我们首先要通过SparseFeat函数指明这些类别型特征， 然后还需要通过VarLenSparseFeat函数再进行序列填充， 使得每个用户的历史序列一样长， 所以这个函数参数中会有个maxlen，来指明序列的最大长度是多少。
   3. 对于连续型特征列， 我们只需要用DenseFeat函数来指明列名和维度即可。
2. 处理完特征列之后， 我们把相应的数据与列进行对应，就得到了最后的数据。

- 更多详细内容：https://tianchi.aliyun.com/notebook/144454