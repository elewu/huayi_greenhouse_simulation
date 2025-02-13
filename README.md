# huayi_greenhouse_simulation

#### 介绍
草莓温室仿真环境


#### 使用方法

1.  修改至自己工作区路径并加入环境变量（可以粘贴进.bashrc文件）

```
    export GAZEBO_RESOURCE_PATH=/home/why/ROS_Projects/2023_plant/src/mybot_description:$GAZEBO_RESOURCE_PATH
    export GAZEBO_MODEL_PATH=/home/why/ROS_Projects/2023_plant/src/mybot_description/models:$GAZEBO_MODEL_PATH
```

2.  编译工程
3.  启动仿真

```bash
roslaunch mybot_description display_mybot.launch
```

4. keyboard

```bash
rosrun turtlesim turtle_teleop_key /turtle1/cmd_vel:=/cmd_vel
```




### env setup

```bash
# ros related
pip install rospkg rospy catkin_tools
pip install defusedxml
pip install empy
pip install netifaces

```




#### 说明
1. mybot_control那部分没用到，只用了teleop_key键盘遥控
2. lidar三个：均为10HZ，其中lidar3平扫，lidar1前视，lidar2后视，参数依照锐驰智光Lakibeam 1s设置。话题名：
- /laser1_scan
- /laser2_scan
- /laser3_scan




# add contact sensor

https://blog.csdn.net/qq_45701501/article/details/107334215


# train rl control

```bash
python scripts/train.py -d look -m sac
```




# train pep

1. collect data

```bash
python scripts/collect_data.py
```

2. process data

```bash

```



