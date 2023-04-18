## 安装

```
git clone http://gitlab.aicrowd.com/Mai/ijcai2022-nmmo-starter-kit.git
conda create -n ijcai2022-nmmo python==3.9
conda activate ijcai2022-nmmo
cd ./ijcai2022-nmmo-starter-kit

apt install git-lfs
pip install git+http://gitlab.aicrowd.com/henryz/ijcai2022nmmo.git
pip install -r requirements_tool.txt
pip install -r requirements.txt
```



## 关键文件结构

```
.
├── my-submission           #提交文件夹
│   ├── checkpoints         #用于存放提交的模型
│   ├── submission.py       #提交到服务器的接口，需要载入提交的模型，并需要根据模型结构修改通过观测计算动作的过程
│   └── torchbeast          #模型代码，和训练的基本一样，不同的是Log的info等级不同，提交的不需要所有都log
├── tool.py                 #测试和提交的工具，一键使用
└── training                #训练的文件夹
    ├── maps                #地图资源，运行代码会自动生成
    ├── results             #训练结果，包括模型，学习曲线
    ├── plot.py             #训练结果的可视化工具，一键使用
    ├── torchbeast          #模型代码，需要在提交的时候和submission里的同步
    └── train.sh            #训练脚本，一键使用

```



## 训练

进入training文件夹，并执行训练脚本

```
cd training
./train.sh
```

等待训练结束



## 测试和提交

测试和提交都需要将模型放到`my-submission`文件夹

* 将打算提交的模型从`training/results/`放到`my-submission/checkpoints`

* 修改`submission.py `中的导入模型的代码

  ```
  class MonobeastBaselineTeam(Team): #根据模型结构修改通过观测计算动作的过程
  	pass
  
  class Submission:
      team_klass = MonobeastBaselineTeam
      init_params = {
          "checkpoint_path":
          Path(__file__).parent / "checkpoints" / "model_113016832.pt"  #导入模型参数
      }
  ```

* 将训练的代码`training/torchbeast`文件夹完全同步到`my-submission/torchbeast`（如果有修改），并将`monobeast.py`文件中的log等级修改为`CRITICAL`

  ```
  #* for submit
  logging.basicConfig(
      format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] "
              "%(message)s"),
      level=logging.CRITICAL,
  )
  
  #* for train
  # logging.basicConfig(
  #     format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] "
  #             "%(message)s"),
  #     level=logging.INFO,
  # )
  ```

### 进行测试

```
python tool.py test
```

### 进行提交

```
python tool.py submit <unique-submission-name>
```

如果可以看到下图，则提交成功

```
        #///(            )///#
         ////      ///      ////
        /////   //////////   ////
        /////////////////////////
     /// /////////////////////// ///
   ///////////////////////////////////
  /////////////////////////////////////
    )////////////////////////////////(
     /////                      /////
   (///////   ///       ///    //////)
  ///////////    ///////     //////////
(///////////////////////////////////////)
          /////           /////
            /////////////////
               ///////////
```

