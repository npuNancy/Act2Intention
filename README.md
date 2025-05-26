
<h1 align= "center"> Act2Intention </h1>


<p align="center">
  <a href="#overview">Overview</a>  | 🤗
  <a href="https://huggingface.co/datasets/BBKKBKK0nancy/Act2Intention">Data Release</a> 
</p>

## Overview

<br>
<div align="center">
<img src="assets/overview.png" width="800px">
</div>
<br>

## Requirements

```bash
pip install -r requirements.txt
```

## Data preparation

Download the dataset [Act2Intention-Bench](https://huggingface.co/datasets/BBKKBKK0nancy/Act2Intention) into `data/trajectory`.

```
python scripts/generate_stage_2.py
python scripts/generate_stage_3.py
```

## Simulator 

### Prepare
- prepare the user intentions. 
- copy `.env.local` to `.env` and set the environment variables.
- prepare the Executor, such as [CogAgent](https://github.com/THUDM/CogAgent), [MobileAgent](https://github.com/X-PLUG/MobileAgent), [UI-TARS](https://github.com/bytedance/UI-TARS), etc.

### Run
1. generate persona: `python simulator\generate_persona.py`
2. generate intention trajectory: `simulator\generate_intention_trajectory.py`
3. generate action trajectory by Executors.

## Train

We train the Agent by [LLamaFactory](https://github.com/hiyouga/LLaMA-Factory)
