<div align="center">
<img width="400px" height="200px" src="https://github.com/FL33TW00D/steelix/raw/master/.github/images/SteelixLogo.png">
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

Steelix is your one stop shop for ONNX model analysis.

## Features
Steelix has 2 main functions:

1. Model Summarization
Steelix can produce an output like below for any provided ONNX model.


2. Graph Visualization
Steelix parses your ONNX file and can transpile it to a DOT file for your
viewing pleasure. This SVG can be embedded in a notebook, website etc. Steelix
runs a mock forward pass through your network to ensure that every edge will
have a shape label!

## How to use

```
steelix plot --model-path ./my-model.onnx --output-path ./my-svg.svg
steelix summary --model-path ./my-model.onnx
```

## Why not Netron?

Netron is a great model visualizer and supports a much wider range of formats that we do. However, there are a few usecases where Steelix excels:

1. Composition: As a CLI tool Steelix is perfect for inclusion inside your workflow/pipelines. Now every time you make an alteration to your model, you can generate a new SVG and embed it!
```
import subprocess
result = subprocess.run(
    ["steelix", "steelix plot --model-path ./my-model.onnx --output-path ./my-svg.svg"], capture_output=True, text=True
) 
```
2. Guaranteed Shapes
3. Metrics

Steelix not only allows you to visualize your model, but can give you handy stats like:
- Total FLOPs
- Total Parameters

4. Guaranteed Shapes
When working with neural nets, the most common problem is mismatched sizes between operations. As Steelix runs a shape pass through the model - we will always have shapes available - even if they aren't included in the ONNX file.

## Install

### Prerequesties
Ensure you have DOT installed by following the instructions [here](https://graphviz.org/download/).

### MacOS
```
brew install steelix
```

### If you already have Rust installed 
```
cargo install steelix
```

## Supported Operators (ref [ONNX IR](https://github.com/onnx/onnx/blob/master/docs/Operators.md?plain=1)) 

| **Operator**              | **Implemented**                      |
|---------------------------|--------------------------------------|
| Abs                       |                                      |
| Acos                      |                                      |
| Acosh                     |                                      |
| Add                       | ✅                                   |
| And                       |                                      |
| ArgMax                    |                                      |
| ArgMin                    |                                      |
| Asin                      |                                      |
| Asinh                     |                                      |
| Atan                      |                                      |
| Atanh                     |                                      |
| AveragePool               |                                      |
| BatchNormalization        | ✅                                   |
| BitShift                  |                                      |
| Cast                      |                                      |
| Ceil                      |                                      |
| Clip                      |                                      |
| Compress                  |                                      |
| Concat                    | ✅                                   |
| ConcatFromSequence        |                                      |
| Constant                  |                                      |
| ConstantOfShape           |                                      |
| Conv                      | ✅                                   |
| ConvInteger               |                                      |
| ConvTranspose             |                                      |
| Cos                       |                                      |
| Cosh                      |                                      |
| CumSum                    |                                      |
| DepthToSpace              |                                      |
| DequantizeLinear          |                                      |
| Det                       |                                      |
| Div                       |                                      |
| Dropout                   |                                      |
| Einsum                    |                                      |
| Elu                       |                                      |
| Equal                     |                                      |
| Erf                       |                                      |
| Exp                       |                                      |
| Expand                    |                                      |
| EyeLike                   |                                      |
| Flatten                   |                                      |
| Floor                     |                                      |
| GRU                       |                                      |
| Gather                    | ✅                                   |
| GatherElements            |                                      |
| GatherND                  |                                      |
| Gemm                      | ✅                                   |
| GlobalAveragePool         |                                      |
| GlobalLpPool              |                                      |
| GlobalMaxPool             |                                      |
| Greater                   |                                      |
| GridSample                |                                      |
| HardSigmoid               |                                      |
| Hardmax                   |                                      |
| Identity                  |                                      |
| If                        |                                      |
| InstanceNormalization     |                                      |
| IsInf                     |                                      |
| IsNaN                     |                                      |
| LRN                       |                                      |
| LSTM                      |                                      |
| LeakyRelu                 |                                      |
| Less                      |                                      |
| Log                       |                                      |
| Loop                      |                                      |
| LpNormalization           |                                      |
| LpPool                    |                                      |
| MatMul                    | ✅                                   |
| MatMulInteger             |                                      |
| Max                       |                                      |
| MaxPool                   | ✅                                   |
| MaxRoiPool                |                                      |
| MaxUnpool                 |                                      |
| Mean                      |                                      |
| Min                       |                                      |
| Mod                       |                                      |
| Mul                       | ✅                                   |
| Multinomial               |                                      |
| Neg                       |                                      |
| NonMaxSuppression         |                                      |
| NonZero                   |                                      |
| Not                       |                                      |
| OneHot                    |                                      |
| Optional                  |                                      |
| OptionalGetElement        |                                      |
| OptionalHasElement        |                                      |
| Or                        |                                      |
| PRelu                     |                                      |
| Pad                       | ✅ (mode=constant, pads>=0)          |
| Pow                       |                                      |
| QLinearConv               |                                      |
| QLinearMatMul             |                                      |
| QuantizeLinear            |                                      |
| RNN                       |                                      |
| RandomNormal              |                                      |
| RandomNormalLike          |                                      |
| RandomUniform             |                                      |
| RandomUniformLike         |                                      |
| Reciprocal                |                                      |
| ReduceL1                  |                                      |
| ReduceL2                  |                                      |
| ReduceLogSum              |                                      |
| ReduceLogSumExp           |                                      |
| ReduceMax                 |                                      |
| ReduceMean                |                                      |
| ReduceMin                 |                                      |
| ReduceProd                |                                      |
| ReduceSum                 |                                      |
| ReduceSumSquare           |                                      |
| Relu                      | ✅                                   |
| Reshape                   | ✅                                   |
| Resize                    |                                      |
| ReverseSequence           |                                      |
| RoiAlign                  |                                      |
| Round                     |                                      |
| Scan                      |                                      |
| Scatter (deprecated)      |                                      |
| ScatterElements           |                                      |
| ScatterND                 |                                      |
| Selu                      |                                      |
| SequenceAt                |                                      |
| SequenceConstruct         |                                      |
| SequenceEmpty             |                                      |
| SequenceErase             |                                      |
| SequenceInsert            |                                      |
| SequenceLength            |                                      |
| Shape                     |                                      |
| Shrink                    |                                      |
| Sigmoid                   | ✅                                   |
| Sign                      |                                      |
| Sin                       |                                      |
| Sinh                      |                                      |
| Size                      |                                      |
| Slice                     |                                      |
| Softplus                  |                                      |
| Softsign                  |                                      |
| SpaceToDepth              |                                      |
| Split                     |                                      |
| SplitToSequence           |                                      |
| Sqrt                      |                                      |
| Squeeze                   | ✅                                   |
| StringNormalizer          |                                      |
| Sub                       |                                      |
| Sum                       |                                      |
| Tan                       |                                      |
| Tanh                      |                                      |
| TfIdfVectorizer           |                                      |
| ThresholdedRelu           |                                      |
| Tile                      |                                      |
| TopK                      |                                      |
| Transpose                 |                                      |
| Trilu                     |                                      |
| Unique                    |                                      |
| Unsqueeze                 | ✅                                   |
| Upsample (deprecated)     |                                      |
| Where                     |                                      |
| Xor                       |                                      |
| **Function**              |                                      |
| Bernoulli                 |                                      |
| CastLike                  |                                      |
| Celu                      |                                      |
| DynamicQuantizeLinear     |                                      |
| GreaterOrEqual            |                                      |
| HardSwish                 |                                      |
| LessOrEqual               |                                      |
| LogSoftmax                |                                      |
| MeanVarianceNormalization |                                      |
| NegativeLogLikelihoodLoss |                                      |
| Range                     |                                      |
| Softmax                   | ✅                                   |
| SoftmaxCrossEntropyLoss   |                                      |

