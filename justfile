plot:
    cargo run --release --bin steelix -- plot --model-path ./resources/models/efficientnet-lite4-11.onnx --output-path ./plot.svg --infer-shapes 
summary:
    cargo run --release --bin steelix -- summary --model-path ./resources/models/efficientnet-lite4-11.onnx
flamegraph:
    sudo cargo flamegraph --bin steelix -- plot --model-path ./resources/models/ResNet101-DUC-12.onnx --output-path ./plot.svg

