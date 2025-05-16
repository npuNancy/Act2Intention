
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

We train the Agent by [LLamaFactory](https://github.com/hiyouga/LLaMA-Factory)

## Data preparation

Download the dataset [Act2Intention-Bench](https://huggingface.co/datasets/BBKKBKK0nancy/Act2Intention) into `data/trajectory`.

```
python scripts/generate_stage_2.py
python scripts/generate_stage_3.py
```