plot:
    cargo run --release --bin steelix -- plot --model-path ./resources/models/efficientnet-lite4-11.onnx --output-path ./plot.svg 
summary:
    cargo run --release --bin steelix -- summary --model-path ./resources/models/efficientnet-lite4-11.onnx
