use clap::ArgMatches;
use ir::{IntoArcTensor, Tensor};
use parser::parse_model;
use smallvec::smallvec;
use std::{collections::HashMap, process::Command as ProcessCommand, sync::Arc};
use steelix::{
    build_cli, hardware_table, metrics_table, opcount_table, render_to, RenderableGraph,
};
use tabled::{
    builder::Builder, col, object::Rows, row, Alignment, Disable, Modify, Panel, Style, Table,
    Tabled,
};
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
    let model_path = matches
        .get_one::<String>("MODEL_PATH")
        .expect("Failed to find model at path.")
        .into();
    let output_path = matches
        .get_one::<String>("OUTPUT_PATH")
        .expect("Invalid output path provided.");

    let plottable: RenderableGraph = parse_model(model_path)
        .expect("Failed to parse model.")
        .into();

    let mut f = NamedTempFile::new().unwrap();
    render_to(&mut f, plottable);
    ProcessCommand::new("dot")
        .arg("-Tsvg")
        .arg(f.path())
        .arg("-o")
        .arg(output_path)
        .output()
        .expect("Failed to call Dot, is it installed?");
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

    let inputs: HashMap<String, Arc<Tensor>> = HashMap::from([(
        "images:0".into(),
        Tensor::new(ir::DType::F32, smallvec![1, 3, 224, 224]).into_arc_tensor(),
    )]);

    let summary = parse_model(model_path)?
        .build_traversal_order()
        .run(inputs)?;

    let op_counts = summary.op_counts.clone();
    let flops = summary.total_flops;

    let summary = vec![
        SummaryTable {
            table: "Operations".to_string(),
            subtable: opcount_table(op_counts),
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
        .with(Panel::header("Model Summary"))
        .with(Disable::row(Rows::single(1)))
        .with(Style::modern())
        .with(Modify::new(Rows::first()).with(Alignment::center()))
        .with(Modify::new(Rows::new(1..)).with(Alignment::left()));

    println!("{}", res);

    Ok(())
}
