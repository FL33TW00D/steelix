plot:
    cargo run --release --bin steelix -- plot --model-path ./resources/models/bvlcalexnet-12.onnx --open 
summary:
    cargo run --release --bin steelix -- summary --model-path ./resources/models/bvlcalexnet-12.onnx
flamegraph:
    cargo flamegraph --root --bin steelix -- plot --model-path ./resources/models/efficientnet-lite4-11.onnx --output-path ./plot.svg

