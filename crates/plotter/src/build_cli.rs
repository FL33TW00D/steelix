use clap::{Arg, Command};

pub fn build_cli() -> Command<'static> {
    let plot_subcommand = Command::new("plot")
        .about("Plotter to plot ONNX files as SVG")
        .arg_required_else_help(true)
        .arg(
            Arg::new("MODEL_PATH")
                .short('m')
                .long("model-path")
                .help("Path to ONNX file to be analyzed.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::new("OUTPUT_PATH")
                .short('o')
                .long("output-path")
                .help("Path where the SVG will be created")
                .takes_value(true)
                .required(true),
        );

    let infer_subcommand = Command::new("infer")
        .about("CPU only ONNX inference")
        .arg_required_else_help(true)
        .arg(
            Arg::new("MODEL_PATH")
                .long("model-path")
                .help("Path to ONNX file for inference.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::new("IMAGE_PATH")
                .long("image-path")
                .help("Path to image to perform inference on.")
                .takes_value(true)
                .conflicts_with("WEBCAM")
                .required(false),
        )
        .arg(
            Arg::new("WEBCAM")
                .long("webcam")
                .help("Launch webcam and perform inference on image stream.")
                .conflicts_with("IMAGE_PATH")
                .required(false),
        );

    Command::new("steelix")
        .subcommand(plot_subcommand)
        .subcommand(infer_subcommand)
        .subcommand_required(true)
        .arg_required_else_help(true)
}
