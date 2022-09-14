<div align="center">
<img width="400px" height="200px" src="https://github.com/FL33TW00D/steelix/raw/master/.github/SteelixLogo.png">
</div>
<p align="center">
    <a href="https://github.com/FL33TW00D/steelix/actions">
        <img alt="Build" src="https://github.com/FL33TW00D/steelix/workflows/ci/badge.svg">
    </a>
    <a href="https://crates.io/crates/steelix">
        <img alt="GitHub" src="https://img.shields.io/github/license/FL33TW00D/steelix.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/transformers/index">
        <img alt="crates.io" src="https://img.shields.io/crates/v/steelix.svg">
    </a>
    <a href="https://github.com/FL33TW00D/steelix/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/FL33TW00D/steelix.svg">
    </a>
</p>

Steelix is a fast CLI based visualizer for ONNX machine learning models! Steelix renders your ONNX model to an SVG using DOT.

## How to use

```
steelix plot --model-path ./my-model.onnx --output-path ./my-svg.svg
```

## Why not Netron?

Netron is a great model visualizer, but it isn't **composeable**. With Steelix, you can add a simple command like below:
```
import subprocess
result = subprocess.run(
    ["steelix", "steelix plot --model-path ./my-model.onnx --output-path ./my-svg.svg"], capture_output=True, text=True
) 
```
straight into your model flow, and view your changes instantly! Steelix is also fast, able to render large models like DEIT in under a second.

## Install

### Prerequesties
Ensure you have DOT installed by following the instructions [here](https://graphviz.org/download/).

### MacOS
```
brew install steelix
```

### Rust Programmers
```
cargo install steelix
```

