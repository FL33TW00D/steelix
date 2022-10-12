use human_repr::HumanCount;
use std::collections::HashMap;

use ir::{DType, ModelSummary};
use tabled::{object::Rows, Alignment, Modify, Panel, Style, Table, Tabled};

use crate::load_devices;

#[derive(Tabled)]
#[tabled(rename_all = "PascalCase")]
struct CountTableEntry {
    op_name: String,
    count: usize,
}

pub fn opcount_table(op_counts: HashMap<String, usize>) -> Table {
    let mut counts = op_counts
        .iter()
        .map(|(k, v)| CountTableEntry {
            op_name: k.to_string(),
            count: *v,
        })
        .collect::<Vec<CountTableEntry>>();
    counts.sort_by(|a, b| b.count.cmp(&a.count));

    let total = counts.iter().fold(0, |acc, count| acc + count.count);

    Table::new(&counts)
        .with(Style::modern())
        .with(Modify::new(Rows::first()).with(Alignment::center()))
        .with(Modify::new(Rows::new(1..)).with(Alignment::left()))
        .with(Panel::footer(format!("{} nodes", total)))
        .to_owned()
}

#[derive(Tabled)]
#[tabled(rename_all = "PascalCase")]
struct MetricsEntry {
    metric: String,
    total: String,
}

pub fn metrics_table(model_summary: ModelSummary) -> Table {
    let metrics = vec![
        MetricsEntry {
            metric: "FLOPS".to_string(),
            total: model_summary.total_flops.human_count_bare().to_string(),
        },
        MetricsEntry {
            metric: "Parameters".to_string(),
            total: model_summary.total_params.human_count_bare().to_string(),
        },
    ];
    Table::new(metrics)
        .with(Style::modern())
        .with(Modify::new(Rows::first()).with(Alignment::center()))
        .with(Modify::new(Rows::new(1..)).with(Alignment::left()))
        .to_owned()
}

#[derive(Tabled)]
#[tabled(rename_all = "PascalCase")]
struct HardwareEntry {
    name: String,
    its: String,
}

pub fn hardware_table(total_flops: usize) -> Table {
    let devices = load_devices().expect("Failed to load devices.");

    let hardware: Vec<HardwareEntry> = devices
        .iter()
        .map(|device| HardwareEntry {
            name: device.name.clone(),
            its: device
                .calculate_its(DType::F32, total_flops)
                .expect("Failed to calculate iterations.")
                .to_string(),
        })
        .collect();

    Table::new(hardware)
        .with(Style::modern())
        .with(Modify::new(Rows::first()).with(Alignment::center()))
        .with(Modify::new(Rows::new(1..)).with(Alignment::left()))
        .to_owned()
}
