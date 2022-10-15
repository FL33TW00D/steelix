<div align="center">
<img width="400px" height="200px" src="https://github.com/FL33TW00D/steelix/raw/master/.github/images/SteelixLogo.png">
</div>
<p align="center">Your one stop CLI for <b>ONNX</b> model analysis. <br></br> Featuring graph visualization, FLOP counts, memory metrics and more!</p>
<p align="center">
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

## ⚡️ Quick start
First, [download](https://graphviz.org/download/) and install DOT.
Installation can be done via `cargo`:

```bash
cargo install steelix
```
MacOS users can also install via HomeBrew:

```bash
brew install steelix
```

## ⚙️ Commands & Options
Steelix has 2 core functions - model summarization & model visualization.

### `summary`

CLI command to summarize the core aspects of your model.

```bash
steelix summary --model-path ./my-model.onnx
```
|     Option       |                       Description                        | Type   | Default | Required? |
|------------------|----------------------------------------------------------|--------|---------|-----------|
| `--model-path`   |             Path at which your model is located.         | `bool` | `false` | No        |


<img width="700px" src="https://github.com/FL33TW00D/steelix/raw/master/.github/images/steelix_summary.gif">

### `plot`

CLI command to plot your model as an SVG file - complete with inferred shapes.

```bash
steelix plot --model-path ./my-model.onnx --open 
```

| Option             | Description                           | Type      | Default       | Required? |
|--------------------|---------------------------------------|-----------|---------------|-----------|
| `--model-path`     | Path at which your model is located.  | `string`  | None          | Yes       |
| `--output-path`    | Path at which your SVG will be saved. | `string`  | `./model.svg` | No        |
| `--open`           | Open SVG in browser once generated.   | `boolean` | `false`       | No        |
| `--disable-shapes` | Disable shape inference.              | `boolean` | `false`       | No        |

<img width="700px" src="https://github.com/FL33TW00D/steelix/raw/master/.github/images/steelix_plot.gif">


## Supported Operators (ref [ONNX IR](https://github.com/onnx/onnx/blob/master/docs/Operators.md?plain=1)) 

| **Operator**              | **Implemented**                      |
|---------------------------|--------------------------------------|
| Abs                       | ✅                                   |
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
| Erf                       | ✅                                   |
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
| LeakyRelu                 | ✅                                   |
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
| Not                       | ✅                                   |
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


## Credit
Most of the good ideas/code in this project are **heavily** inspired by [tract](https://github.com/sonos/tract), [wonnx](https://github.com/webonnx/wonnx) or [netron](https://github.com/lutzroeder/netron).
