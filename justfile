plot:
    cargo run --release --bin steelix -- plot --model-path ./resources/models/efficientnet-lite4-11-int8.onnx --output-path ./plot.svg --infer-shapes 
summary:
    cargo run --release --bin steelix -- summary --model-path ./resources/models/efficientnet-lite4-11-int8.onnx
flamegraph:
    cargo flamegraph --root --bin steelix -- plot --model-path ./resources/models/efficientnet-lite4-11.onnx --output-path ./plot.svg

