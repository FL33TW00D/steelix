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
        .subcommand(plot_subcommand)
        .subcommand(summary_command)
        .subcommand_required(true)
        .arg_required_else_help(true)
}