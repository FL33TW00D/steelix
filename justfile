plot:
    cargo run --release --bin steelix -- plot --model-path ./resources/models/bvlcalexnet-12.onnx --output-path ./plot.svg --infer-shapes 
flamegraph:
    sudo cargo flamegraph --bin steelix -- plot --model-path ./resources/models/ResNet101-DUC-12.onnx --output-path ./plot.svg
summary:
    cargo run --release --bin steelix -- summary --model-path ./resources/models/bvlcalexnet-12.onnx

