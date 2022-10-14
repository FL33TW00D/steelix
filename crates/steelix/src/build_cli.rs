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
            Arg::new("INFER_SHAPES")
                .short('s')
                .long("infer-shapes")
                .takes_value(false)
                .help("Infer shapes of the model."),
        )
        .arg(
            Arg::new("OPEN_IN_BROWSER")
                .long("open")
                .takes_value(false)
                .help("Open the SVG in the browser."),
        )
        .arg(
            Arg::new("OUTPUT_PATH")
                .short('o')
                .long("output-path")
                .help("Path where the SVG will be created")
                .default_value("model.svg")
                .takes_value(true),
        );

    let summary_command = Command::new("summary")
        .about("Summary of model operations and their cost")
        .arg_required_else_help(true)
        .arg(
            Arg::new("MODEL_PATH")
                .long("model-path")
                .help("Path to ONNX file for inference.")
                .takes_value(true)
                .required(true),
        );

    Command::new("steelix")
        .about("ONNX model analyzer")
        .long_about(
            "Steelix is a tool to analyze ONNX models and provide insights into their structure and \
             performance.",
        )
        .subcommand(plot_subcommand)
        .subcommand(summary_command)
        .subcommand_required(true)
        .arg_required_else_help(true)
}
