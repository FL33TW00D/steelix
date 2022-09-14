use clap::ArgMatches;
use parser::parse_model;
use std::process::Command as ProcessCommand;
use steelix::{build_cli, render_to, RenderableGraph};
use tempfile::NamedTempFile;

fn main() {
    let matches = build_cli().get_matches();
    match matches.subcommand().unwrap() {
        ("plot", matches) => run_plot_command(matches),
        _ => unreachable!("Invalid command provided."),
    }
}

fn run_plot_command(matches: &ArgMatches) {
    let model_path = matches
        .get_one::<String>("MODEL_PATH")
        .expect("Failed to find model at path.")
        .into();
    let output_path = matches
        .get_one::<String>("OUTPUT_PATH")
        .expect("Invalid output path provided.");

    let plottable: RenderableGraph = parse_model(model_path).unwrap().into();

    let mut f = NamedTempFile::new().unwrap();
    render_to(&mut f, plottable);
    ProcessCommand::new("dot")
        .arg("-Tsvg")
        .arg(f.path())
        .arg("-o")
        .arg(output_path)
        .output()
        .expect("Failed to call Dot, is it installed?");
}
