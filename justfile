plot:
    cargo run --release --bin steelix -- plot --model-path ./resources/models/bvlcalexnet-12.onnx --output-path ./plot.svg 
summary:
    cargo run --release --bin steelix -- summary --model-path ./resources/models/mobilenetv2-7-sim.onnx
