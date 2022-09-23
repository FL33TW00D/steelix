plot:
    cargo run --release --bin steelix -- plot --model-path ./resources/models/MaskRCNN-10-sim.onnx --output-path ./plot.svg 
summary:
    cargo run --release --bin steelix -- summary --model-path ./resources/models/MaskRCNN-10-sim.onnx

