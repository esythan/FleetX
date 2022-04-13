流式训练
=====================

简介
---------------------
飞桨参数服务器训练支持流式训练模式，支持配置千亿级大规模稀疏及[0, INT64]范围内的ID映射，支持模型自增长及配置特征准入（不存在的特征可以以适当的条件创建）、淘汰（够以一定的策略进行过期的特征的清理）等策略，支持模型增量保存，通过多种优化来保证流式训练的流程及效果。

本章节只介绍流式训练需要的重点API和配置，具体流式训练示例请参考PaddleRec中的\ `流式训练 <https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/online_trainer.md>`_\内容。


原理介绍
---------------------
流式训练(OnlineLearning)， 即训练数据不是一次性放入训练系统，而是随着时间流式的加入到训练过程中去。 整个训练服务不停止，数据经过预处理后进入训练系统参与训练并产出线上所需的预测模型参数。通过流式数据的生产、实时训练及快速部署上线来提升推荐系统的性能和效果。流式训练是按照一定顺序进行数据的接收和处理，每接收一个数据，模型会对它进行预测并对当前模型进行更新，然后处理下一个数据。 像信息流、小视频、电商等场景，每天都会新增大量的数据， 让每天(每一刻)新增的数据基于上一天(上一刻)的模型进行新的预测和模型更新。


功能效果
---------------------
通过合理配置，可实现大规模流式训练，提升推荐系统的性能和效果。

本文中涉及到的相关功能和使用示例：

- 使用大规模稀疏的算子进行组网
- 配置准入策略
- 配置模型保存及增量保存


使用方法
---------------------

大规模稀疏及准入配置
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: paddle.static.nn.sparse_embedding(input, size, padding_idx=None, is_test=False, entry=None, param_attr=None, dtype='float32')

飞桨参数服务器模式使用使用`paddle.static.nn.sparse_embedding`作为embedding lookup层的算子， 而不是使用 `paddle.nn.functional.embedding`。
`paddle.static.nn.sparse_embedding` 采用稀疏模式进行梯度的计算和更新，输入接受[0, INT64]范围内的特征ID,  更加符合在线学习的功能需求。


**参数：**

    - input, (Tensor): 存储特征ID的Tensor，数据类型必须为：int32/int64，特征的范围在[0, INT64]之间，超过范围在运行时会提示错误。
    - size, (list|tuple): 形状为(num_embeddings, embedding_dim), 大规模稀疏场景下， 参数规模初始为0，会随着训练的进行逐步扩展，因此num_embeddings 暂时无用，可以随意填一个整数，embedding_dim 则为词嵌入权重参数的维度配置。
    - padding_idx, (int): 如果配置了padding_idx，那么在训练过程中遇>到此id时会被用0填充。
    - is_test, (bool): 训练/预测模式，在预测模式(is_test=False)下，遇到不存在的特征，不会初始化及创建，直接以0填充后返回。
    - entry, (ShowClickEntry, optinal): 准入策略配置，支持用户自定义传入稀疏特征的展现(show)和点击(click)次数，稀疏参数的embedding会有两个维度统计特征的总展现和总点击量，用于稀疏参数的准入、淘汰、保存等。
    - param_attr, (ParamAttr, optinal): embedding层参数属性，类型是ParamAttr或者None， 默认为None。
    - dtype, (float32|float64, optinal): 输出Tensor的数据类型，支持float32、float64。当该参数值为None时， 输出Tensor的数据类型为float32。默认值为None。

**用法示例：**

    .. code:: python

        ...

        import paddle

        sparse_feature_dim = 1024
        embedding_size = 64

        # 训练过程中，出现超过10次及以上的特征才会参与训练
        # 构造ShowClickEntry，指明展现和点击对应的变量名
        entry = paddle.distributed.ShowClickEntry("show", "click")

        # 构造show/click对应的data，变量名需要与entry中的名称一致
        show = paddle.static.data(
            name="show", shape=[None, 1], dtype="int64")
        label = paddle.static.data(
            name="click", shape=[None, 1], dtype="int64")

        input = paddle.static.data(name='ins', shape=[1], dtype='int64')

        emb = paddle.static.nn.sparse_embedding((
            input=input,
            size=[sparse_feature_dim, embedding_size],
            is_test=False,
            entry=entry,
            param_attr=paddle.ParamAttr(name="SparseFeatFactors",
                    initializer=paddle.nn.initializer.Uniform()))

飞桨参数服务器模式下的稀疏参数，存储于PServer端的SparseTable中，并使用Accessor对SparseTable中的参数进行更新、保存、淘汰等。

用户可以通过对Accessor进行配置来影响稀疏参数的各种功能：

配置说明参见PaddleRec中的\ `流式训练 <https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/online_trainer.md>`_\中的高级功能章节。

具体配置内容参加PaddleRec的slot_dnn模型示例的\ `config_online.yaml <https://github.com/PaddlePaddle/PaddleRec/blob/master/models/rank/slot_dnn/config_online.yaml>`_\的table_parameters部分。

稀疏参数淘汰
~~~~~~~~~~

.. py:function:: paddle.distributed.fleet.shrink()

使用此接口，可以对长久不出现或者出现频率极低的ID特征进行清理。
稀疏参数在初始化的时候，会在内部设定一个unseen_day，记录该ID未出现的天数，当值超过Accessor配置中的`delete_after_unseen_days`时，则会被清理掉。
同时，Accessor会利用SparseTable中保存的稀疏参数统计量（show/click）计算特征的频次score，当该score值小于Accessor配置中的`delete_threshold`时，也会被清理掉。


    .. code:: python

        ...

        import paddle

        ...
        dataset, hour, day = get_ready_training_dataset()

        do_training ...

        # 天级别的淘汰，每天的数据训练结束后，对长久不出现或者出现频率极低的ID特征进行清理
        if fleet.is_first_worker() and hour == 23:
            paddle.distributed.fleet.shrink()



保存及增量保存配置
~~~~~~~~~~~~~~~~~~~~~
具体API接口参考\ `增量训练 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_incremental_learning.html>`_\保存模型相关部分。
一般在流式训练中，会保存两种类型的模型：

- checkpoint模型，用于在训练异常中断后，全量加载模型并继续增量训练，由于该模型占用存储过大，一般保存频率较低。
- inference模型，用于线上部署，base+delta模型，base模型一般每天保存一次，然后以base模型为基础，每隔一段时间保存一个delta模型。

    .. code:: python

        ...

        import paddle

        ...
        dataset, hour, day = get_ready_training_dataset()

        do_training ...

        save_delta_freq = 1    # 保存delta模型的频率
        save_checkpoint_freq = 2   # 保存checkpoint模型的频率

        if fleet.is_first_worker() and hour == 0:
            # 每天的0点，保存一次checkpoint模型和base模型
            if hour == 0:
                # 保存checkpoint模型
                fleet.save_persistables(exe, model_path, mode=0)
                # 保存base模型
                fleet.save_inference_model(exe, model_path, feed_var_names, target_vars, mode=2)

            else:
                if hour % save_delta_freq == 0:
                    # 每一小时保存一个delta模型
                    fleet.save_inference_model(exe, model_path, feed_var_names, target_vars, mode=1)
                if hour % save_checkpoint_freq == 0:
                    # 每两小时保存一个checkpoint模型
                    fleet.save_persistables(exe, model_path, mode=0)


常规训练流程
~~~~~~~~~~~~~~~~~~~~~

流式训练是个上下游牵涉众多的训练方法，本文只贴出训练相关的配置给用户做一个讲解。
完整的流式训练示例可参考\ `流式训练脚本 <https://github.com/PaddlePaddle/PaddleRec/blob/master/tools/static_ps_online_trainer.py>`_\，并结合自己的业务需求进行修改。

.. code-block:: python

    # 初始化分布式环境
    fleet.init()

    # your real net function
    model = net()

    # 使用参数服务器异步训练模式
    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.a_sync = True

    # 分布式训练图优化
    adam = paddle.optimizer.Adam(learning_rate=5e-06)
    adam = fleet.distributed_optimizer(adam, strategy=strategy)
    adam.minimize(model.avg_cost)

    # 启动PServer
    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()

    if fleet.is_worker():
        # 初始化Worker
        exe.run(paddle.static.default_startup_program())
        fleet.init_worker()

        while True:
            # 持续不断的从`get_ready_training_set`获取可训练的书记集和相关的配置
            # 下面是一个按小时训练的例子
            dataset, hour, day = get_ready_training_dataset()

            if dataset is None:
                break

            # 使用`dataset`中的数据进行训练和模型保存
            exe.train_from_dataset(program=paddle.static.default_main_program(),
                                   dataset=dataset,
                                   fetch_list=[model.auc],
                                   fetch_info=["avg_auc"],
                                   print_period=10)

            if hour == 23:
                paddle.distributed.fleet.shrink()

            if fleet.is_first_worker() and hour == 0:
                # 每天的0点，保存一次checkpoint模型和base模型
                if hour == 0:
                    # 保存checkpoint模型
                    fleet.save_persistables(exe, model_path, mode=0)
                    # 保存base模型
                    fleet.save_inference_model(exe, model_path, feed_var_names, target_vars, mode=2)

                else:
                    if hour % save_delta_freq == 0:
                        # 每一小时保存一个delta模型
                        fleet.save_inference_model(exe, model_path, feed_var_names, target_vars, mode=1)
                    if hour % save_checkpoint_freq == 0:
                        # 每两小时保存一个checkpoint模型
                        fleet.save_persistables(exe, model_path, mode=0)
            fleet.barrier_worker()
        fleet.stop_worker()



运行成功提示
---------------------
[略]


常见问题与注意事项
---------------------
1. 训练过程中，如需使用分布式指标，请参考<分布式指标章节>。
2. 如果训练中途中断，需要加载模型后继续训练，请参考<增量训练章节>


论文/引用
---------------------
[略]

