<div align="center">
<img width="400px" height="200px" src="https://github.com/FL33TW00D/steelix/raw/master/.github/SteelixLogo.png">
</div>

[![Build status](https://github.com/FL33TW00D/steelix/workflows/ci/badge.svg)](https://github.com/FL33TW00D/steelix/actions)
[![Crates.io](https://img.shields.io/crates/v/steelix.svg)](https://crates.io/crates/steelix)
[![Packaging status](https://repology.org/badge/tiny-repos/steelix.svg)](https://repology.org/project/steelix/badges)


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

