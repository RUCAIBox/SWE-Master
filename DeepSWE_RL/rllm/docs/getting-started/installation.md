# Installation Guide

This guide will help you set up rLLM on your system.

## Prerequisites

Before installing rLLM, ensure you have the following:

- Python 3.10 or higher
- CUDA version >= 12.1
- [uv](https://docs.astral.sh/uv/) package manager

## Installing uv

If you don't have uv installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Basic Installation

rLLM uses [verl](https://github.com/volcengine/verl) as its training backend. Follow these steps to install rLLM and our custom fork of verl:

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/agentica-project/rllm.git
cd rllm

# create a conda environment
conda create -n rllm python=3.10
conda activate rllm

# Install all dependencies
pip install -e ./verl
pip install -e .
```

This will install rLLM and all its dependencies in development mode.

For more help, refer to the [GitHub issues page](https://github.com/agentica-project/rllm/issues). 