# intention-recognition
意图识别案例

仓库初始化：

`git clone https://github.com/SpringDuB/intention-recognition.git`

`cd intention-recognition`

训练数据集格式：

csv文件，两列，[文本， 标签]

##### 训练步骤：

 ###### 将训练数据分成两份，一份做训练一份用于训练测试

`sh split_data.sh`

 ###### 开始训练，bert模型路径请自行修改

`sh training.sh`

##### 线上部署：

`python serveer_runner.py`
