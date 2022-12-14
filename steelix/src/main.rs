use clap::ArgMatches;
use std::process::Command as ProcessCommand;
use steelix::{
    build_cli, hardware_table, metrics_table, opcount_table, parse_model, render_to,
    RenderableGraph,
};
use tabled::{object::Rows, Alignment, Disable, Modify, Panel, Style, Table, Tabled};
use tempfile::NamedTempFile;

fn main() {
    let matches = build_cli().get_matches();
    match matches.subcommand().unwrap() {
        ("plot", matches) => run_plot_command(matches).unwrap(),
        ("summary", matches) => run_summary_command(matches).unwrap(),
        _ => unreachable!("Invalid command provided."),
    }
}

fn run_plot_command(matches: &ArgMatches) -> anyhow::Result<()> {
    let model_path = &matches
        .get_one::<String>("MODEL_PATH")
        .expect("Failed to find model at path.")
        .into();
    let output_path = matches
        .get_one::<String>("OUTPUT_PATH")
        .expect("Invalid output path provided.");
    let disable_shapes = matches.is_present("DISABLE_SHAPES");
    let open = matches.is_present("OPEN_IN_BROWSER");

    let model = parse_model(model_path)?;
    let mut model_summary = None;
    if !disable_shapes {
        model_summary = Some(parse_model(model_path)?.build_traversal_order().run()?);
    }
    let plottable: RenderableGraph = RenderableGraph::build_graph(model, model_summary);

    let mut f = NamedTempFile::new().expect("Failed to create temp file.");
    render_to(&mut f, plottable);
    ProcessCommand::new("dot")
        .arg("-Tsvg")
        .arg(f.path())
        .arg("-o")
        .arg(output_path)
        .output()
        .expect("Failed to call Dot, is it installed?");

    if open {
        opener::open(output_path)?;
    }

    Ok(())
}

#[derive(Tabled)]
struct SummaryTable {
    #[tabled(rename = "")]
    table: String,
    #[tabled(rename = "")]
    subtable: Table,
}

fn run_summary_command(matches: &ArgMatches) -> anyhow::Result<()> {
    let model_path = matches
        .get_one::<String>("MODEL_PATH")
        .expect("Failed to find model at path.")
        .into();

    let summary = parse_model(&model_path)?.build_traversal_order().run()?;
    let op_frequencies = summary.op_frequencies.clone();
    let flops = summary.total_flops;

    let summary = vec![
        SummaryTable {
            table: "Operations".to_string(),
            subtable: opcount_table(op_frequencies),
        },
        SummaryTable {
            table: "Metrics".to_string(),
            subtable: metrics_table(summary),
        },
        SummaryTable {
            table: "Hardware".to_string(),
            subtable: hardware_table(flops),
        },
    ];

    let res = Table::new(summary)
        .with(Panel::header(format!(
            "{} Model Summary",
            model_path.file_stem().unwrap().to_str().unwrap()
        )))
        .with(Disable::row(Rows::single(1)))
        .with(Style::modern())
        .with(Modify::new(Rows::new(0..)).with(Alignment::center()))
        .to_owned();

    println!("{}", res);

    Ok(())
}
