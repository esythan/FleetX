增量训练
=====================

简介
---------------------

增量训练是一种常见的机器学习方法，在深度学习领域内有广泛的应用，它代表的是一种动态学习的训练方式，即训练数据随着时间的推移源源不断的加入到当前训练流程中，扩展当前模型的知识和能力。

飞桨的参数服务器训练支持增量训练，支持训练在某一时间点进行训练模型参数(含部分优化器的状态)的全量保存，在重启训练时将之前保存的全量参数进行加载，结合新的训练数据继续训练，从而学习到新的有用信息。


原理介绍
---------------------
飞桨模型参数在参数服务器下分为稠密参数和稀疏参数两种，训练节点分为PServer和Worker两种，每个Worker上都有完整的稠密参数，没有稀疏参数。稀疏参数和稠密参数分布于全部的PServer节点上。

飞桨参数服务器增量训练包含两部分内容，即模型保存和模型加载。其中保存的模型分为以下两种类型：

- checkpoint模型，包含全部模型参数及优化器状态，可在PServer端全量保存和加载。
- inference模型，稀疏参数通过PServer保存和加载，包括稠密参数及优化器状态；稠密参数通过0号Worker保存和加载，仅有稠密参数而无优化器状态（产出二进制文件）。

通常情况下，增量训练基于checkpoint模型，inference模型用于线上部署，飞桨参数服务器提供了非常方便的API来保存和加载这两种类型的模型。

功能效果
---------------------
- 训练开始时，实现模型参数的全量加载。
- 训练结束时，实现模型参数的全量保存。

使用方法
---------------------

保存checkpoint模型
~~~~~~~~~~~~~~~~~
.. py:function:: paddle.distributed.fleet.save_persistables(executor, dirname, main_program=None, mode=0)
飞桨参数服务器模式使用`paddle.distributed.fleet.save_persistables`保存全量的checkpoint模型，包括全部的稀疏参数和稠密参数，在PServer端以分片方式保存

**参数：**
    - executor (Executor): 用于保存模型的executor。
    - dirname (string): 模型的保存路径。
    - main_program (Program): 模型的Program。
    - mode (int): 保存模式，其中0为全量模型。

**用法示例：**

    .. code:: python

        ...

        import paddle.distributed.fleet as fleet

        dirname = "/you/path/to/model"
    
        if fleet.is_first_worker():
            fleet.save_persistables(exe, dirname, mode=0)
    
    

保存inference模型
~~~~~~~~~~~~~~~~
.. py:function:: paddle.distributed.fleet.inference_model(executor, dirname, feeded_var_names, target_vars, main_program=None, mode=0)
飞桨参数服务器模式使用`paddle.distributed.fleet.save_inference_model`保存用于线上部署的模型。

该模型包括三部分：

- 稀疏参数：在PServer端以分片方式保存
- 稠密参数：在0号Worker上保存成线上服务可以加载的二进制文件
- 模型文件：在0号Worker上保存裁剪之后的模型

**参数：**
    - executor (Executor): 用于保存模型的executor。
    - dirname (string): 模型的保存路径。
    - feeded_var_names (list of string): 模型输入var_name，用于模型裁剪
    - target_vars (list of tensor): 模型输出var，用于模型裁剪
    - main_program (Program): 模型的Program。
    - mode (int): 保存模式，其中1为delta模型，2为base模型，在稀疏参数拥有准入配置的情况下，可能会丢一部分未被准入的特征。

**用法示例：**

    .. code:: python

        ...

        import paddle.distributed.fleet as fleet

        dirname = "/you/path/to/model"
    
        if fleet.is_first_worker():
            fleet.save_inference_model(exe,
                                       dirname, 
                                       [feed.name for feed in feed_vars],
                                       target_vars,
                                       mode=1)

模型加载
~~~~~~~
.. py:function:: paddle.distributed.fleet.load_model(path, mode)
飞桨参数服务器模式使用`paddle.distributed.fleet.load_model`加载模型。
**参数：**
    - path (string): 模型的保存路径。
    - mode (int): 加载模式，其中0为checkpoint模型，1为delta模型，2为base模型。
**用法示例：**

    .. code:: python

        ...

        import paddle.distributed.fleet as fleet

        dirname = "/you/path/to/model"
    
        if fleet.is_first_worker():
            fleet.load_model(dirname, mode=0)


load_model()接口可以同时加载全部的稀疏参数和稠密参数，支持checkpoint模型和inference模型。

另外，飞桨参数服务器还提供了另外一种加载模型的方式，针对inference模型，分别加载稀疏参数和稠密参数，并且可以指定需要加载的参数。

训练启动时每个PServer的基本初始流程如下：

- 每个节点执行 `fleet.init_server(dirname=None, var_names=None, **kwargs)` 进行PServer端初始化。 init_server用有两个选配参数，分别是 `dirname`和`var_names`,`dirname`表示需要增量加载的模型路径，`var_names`指定需要加载的稀疏参数名。 注意，`init_server` 只会加载稀疏参数，稠密参数的加载在Worker端进行。
- 每个节点执行 `fleet.run_server()` 表明当前节点已经初始化成功，可以支持Worker端的连接和通信。


训练启动时每个Worker的基本初始流程如下：

- 每个节点执行 `exe.run(paddle.static.default_startup_program())` 进行参数初始化。
- 0号节点执行 `paddle.static.load_vars()` 指定要加载的稠密参数的名字列表和模型目录，将稠密参数通过此方式进行加载。
- 每个节点执行 `fleet.init_worker()` ， 其中0号节点的稠密参数将同步给相应的PServer，其他节点(非0号)会从PServer端将稠密参数取回本地赋值给本地的稠密参数。

.. code-block:: python

    # 模型加载需要区分是PServer还是Worker
    dirname = "/you/path/to/model"
    
    if fleet.is_server():
        sparse_varnames = [var.name for var in get_sparse_vars()]
        fleet.init_server(dirname, sparse_varnames)
        fleet.run_server()

    if fleet.is_worker():
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
    
        exe.run(paddle.static.default_startup_program())
        dense_vars = get_dense_vars()
        paddle.static.load_vars(executor=exe, dirname=path, vars=dense_vars)
        fleet.init_worker()

备注：文章末尾附录了获取稀疏/稠密参数的代码，参考或复制使用。

运行成功提示
---------------------

1. 模型加载当前并没有提示
2. 模型保存成功，会在相应的目录保存下模型文件。


常见问题与注意事项
---------------------

- 节点动态调整
 + 训练节点在发生变化的情况下，只要稀疏参数的分片数不发生改变（shard_num默认为1000），则可以成功加载模型。

- 加载指定稠密参数
 + 用户可以选择性的加载所需的稠密参数，具体是在 0号 Worker 执行 `paddle.static.load_vars`时 ，指定的 vars的列表来控制。

- 加载指定稀疏参数
 + 用户可以选择性的加载指定的稀疏参数，具体是在PServer执行`init_server`时，指定`var_names`的列表，通过此列表来控制加载的参数名单。



论文/引用
---------------------
[略]

附录
------------------

获取稀疏/稠密参数的代码
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

        def get_sparse_vars():
            import paddle
            program = paddle.static.default_main_program()
            SPARSE_OP_TYPE_DICT = {"lookup_table", "lookup_table_v2"}

            def is_sparse_op(op):
                if op.type in SPARSE_OP_TYPE_DICT and op.attr('is_sparse') is True or \
                    op.type == "distributed_lookup_table":
                    return True
                return False

            def get_sparse_varnames():
                tablenames = set()
                for op in program.global_block().ops:
                    if is_sparse_op(op):
                        tablenames.add(op.input("W")[0])
                return list(tablenames)

            varnames = get_sparse_varnames()

            sparse_vars = set()
            for varname in varnames:
                sparse_vars.add(program.global_block().vars[varname])
            return list(sparse_vars)

        def get_dense_vars():
            import paddle
            program = paddle.static.default_main_program()

            def is_persistable(var):
                if var.desc.type() == paddle.fluid.core.VarDesc.VarType.FEED_MINIBATCH or \
                   var.desc.type() == paddle.fluid.core.VarDesc.VarType.FETCH_LIST or \
                   var.desc.type() == paddle.fluid.core.VarDesc.VarType.READER:
                    return False
                return var.persistable

            exe = paddle.static.Executor(paddle.CPUPlace())
            sparse_varnames = [var.name for var in get_sparse_vars()]
            dense_vars = set()
            for name, var in program.global_block().vars.items():
                if is_persistable(var) and var.name not in sparse_varnames:
                    dense_vars.add(var)
            return list(dense_vars)

