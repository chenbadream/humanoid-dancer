# Humanoid Dancer

## 💻 Installation

1. Create environment and install torch

   ```text
   conda create -n dancer python=3.8 
   conda activate dancer
   pip3 install torch torchvision torchaudio 
   ```

2. Install Isaac Gym preview 4 release https://developer.nvidia.com/isaac-gym

   unzip files to a folder, then install with pip:

   `cd isaacgym/python && pip install -e .`

   check it is correctly installed by playing: 

   ```cmd
   cd examples && python 1080_balls_of_solitude.py
   ```

3. Clone this codebase and install `rsl_rl`.

   ```cmd
   cd rsl_rl
   pip install -e .
   ```

4. Install our `legged_gym`

   ```cmd
   cd legged_gym
   pip install -e .
   ```

   Ensure you have installed the following packages:
    + pip install numpy==1.20 (must < 1.24)
    + pip install tensorboard
    + pip install setuptools==59.5.0

6. Install additional packages `requirements.txt`

   ```cmd
   pip install -r requirements.txt
   ```

## 🛠️ Usage

You need to run these commands under the `legged_gym` folder. The parameter `headless` controls whether to enable the GUI; enabling the GUI requires a device with a screen.

Our parameters are managed by a lightweight command-line interface [tyro](https://github.com/brentyi/tyro). When running the following scripts, you can use `-h` to view all parameters and modify them as needed.

- Train a walking policy for H1:

```
python legged_gym/scripts/train.py --max-iterations 10000 env-cfg:h1-cfg train-cfg:h1-cfg        
```

- Train a DeepMimic-styled imitation policy for H1:

```
python legged_gym/scripts/train.py --max-iterations 100000 env-cfg:h1-mimic-cfg --env-cfg.motion.motion-file resources/motions/h1/amass_phc_filtered.pkl train-cfg:h1-mimic-ppo-cfg
```


When running these policies, a `logs` folder will appear, containing task-specific subfolders like `h1` or `h1-mimic`. These are timestamped folders storing training logs and model checkpoints.

- Load and run a policy without GUI:

```
python legged_gym/scripts/play.py --load_run {run_name} --num_envs 1 env-cfg:h1-cfg train-cfg:h1-ppo-cfg 
```

or

```
python legged_gym/scripts/play.py --load_run {run_name} --num_envs 1 env-cfg:h1-mimic-cfg --env-cfg.viewer.debug-viz train-cfg:h1-mimic-ppo-cfg 
```

- Load and run a policy with GUI:

```
python legged_gym/scripts/play.py --load_run {run_name} --no-headless --num_envs 1 env-cfg:h1-cfg train-cfg:h1-ppo-cfg 
```

or

```
python legged_gym/scripts/play.py --load_run {run_name} --no-headless --num_envs 1 env-cfg:h1-mimic-cfg --env-cfg.viewer.debug-viz train-cfg:h1-mimic-ppo-cfg
```

If you enable the GUI, you can use the following keys for interaction:

| Keyboard | Function |
| ---- | --- |
| f | focus on humanoid |
| Right click + WASD | change view port |
| r | reset episode |
| j | apply large force to the humanoid |
| q | last motion |
| e | next motion |

You can search for `viewer_keyboard_event` in the code to learn more.

## 📚 Data

You can find the required motion files and sample training log files for this codebase in the cloud drive [humanoid-dancer](https://cloud.tsinghua.edu.cn/d/caa33771d5ef4f0f9d55/).

For the motion files, place them in the `legged_gym/resources/motions` directory so the program can access them. You can also visualize them using the script:


```
python legged_gym/scripts/vis.py env-cfg:h1-mimic-cfg --env-cfg.motion.sync --env-cfg.viewer.debug_viz
```


These motion files come from [H2O](https://github.com/LeCAR-Lab/human2humanoid/tree/main), and you can explore that codebase to learn more about their usage.

As for log files, you can place them in the previously mentioned location to load and visualize policies, or use TensorBoard to view specific training curves.

## 🔧 Troubleshooting

1. When you run any scripts, it reports

```
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
```

To fix it, you need change your system path by

```
export LD_LIBRARY_PATH=#HOME/miniconda3/envs/dancer/lib::$LD_LIBRARY_PATH
```

2. When you try to use GUI, it reports

```
[Error] [carb.windowing-glfw.plugin] GLFW initialization failed.
[Error] [carb.windowing-glfw.plugin] GLFW window creation failed!
[Error] [carb.gym.plugin] Failed to create Window in CreateGymViewerInternal
```

It means you are using a headless machine. You can only run it under `headless` mode.

## 😺 Acknowledgement

Our code is generally built upon: [Unitree RL GYM](https://github.com/unitreerobotics/unitree_rl_gym/tree/main), [H2O](https://github.com/LeCAR-Lab/human2humanoid), [PHC](https://github.com/ZhengyiLuo/PHC), [ExBody](https://github.com/chengxuxin/expressive-humanoid). We thank all these authors for their nicely open sourced code and their great contributions to the community.

## 🤔 Tutorials

Since this codebase is based on legged gym and has a certain learning curve, here are some video learning resources for reference.

- [强化学习框架-Legged Gym 训练代码详解](https://www.bilibili.com/video/BV1sLx6eLEyt/?vd_source=124f72b97ca551839539ebce94fa3bc4)
- [[上]6个人形双足强化学习开源项目，论文讲解，代码速读，FLD，PBRS，footstep，ExBody，humanplus，humanoid-gym]([上]6个人形双足强化学习开源项目，论文讲解，代码速读，FLD，PBRS，footstep，ExBody，humanplus，humanoid-gym)
- [[下]6个人形双足强化学习开源项目，论文讲解，代码速读，FLD，PBRS，footstep，ExBody，humanplus，humanoid-gym](https://www.bilibili.com/video/BV1xGphe1E84?spm_id_from=333.788.videopod.sections&vd_source=124f72b97ca551839539ebce94fa3bc4)