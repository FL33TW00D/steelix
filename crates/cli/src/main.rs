use clap::ArgMatches;
use parser::parse_model;
use plotting::{render_to, RenderableGraph};
use std::process::Command as ProcessCommand;
use steelix::{build_cli, infer_path, infer_webcam};
use tempfile::NamedTempFile;

fn main() {
    let matches = build_cli().get_matches();
    match matches.subcommand().unwrap() {
        ("plot", matches) => run_plot_command(matches),
        ("infer", matches) => run_infer_command(matches),
        _ => unreachable!("Invalid command provided."),
    }
}

fn run_infer_command(matches: &ArgMatches) {
    let runnable = parse_model(matches.get_one::<String>("MODEL_PATH").unwrap().into());
    if matches.contains_id("WEBCAM") {
        infer_webcam(runnable);
    } else {
        let image_path = matches.get_one::<String>("IMAGE_PATH").unwrap().to_string();
        infer_path(runnable, image_path);
    }
}

fn run_plot_command(matches: &ArgMatches) {
    let plottable: RenderableGraph =
        parse_model(matches.get_one::<String>("MODEL_PATH").unwrap().into()).into();

    let mut f = NamedTempFile::new().unwrap();
    render_to(&mut f, plottable);
    ProcessCommand::new("dot")
        .arg("-Tsvg")
        .arg(f.path())
        .arg("-o")
        .arg(matches.get_one::<String>("OUTPUT_PATH").unwrap())
        .output()
        .expect("Failed to call Dot, is it installed?");
}
