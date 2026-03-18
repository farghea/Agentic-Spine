# Spine Project Setup

This project requires two Conda environments:

## 1. OpenSim Environment (`opensim`)
**Python Version:** 3.11.14

Install using the YAML file:
```bash
conda env create -f environment_opensim.yml
```

*Or manually via requirements:*
```bash
conda create -n opensim python=3.11.14 -y
conda activate opensim
conda install -c opensim-org opensim -y
pip install -r requirements_opensim.txt
```

## 2. Agent Environment (`agent`)
**Python Version:** 3.12.12

Install using the YAML file:
```bash
conda env create -f environment_agent.yml
```

*Or manually via requirements:*
```bash
conda create -n agent python=3.12.12 -y
conda activate agent
pip install -r requirements_agent.txt
```
