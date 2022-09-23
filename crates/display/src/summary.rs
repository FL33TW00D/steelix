use human_repr::HumanCount;
use std::collections::HashMap;

use ir::{DType, ModelSummary};
use tabled::{object::Rows, Alignment, Modify, Style, Table, Tabled};

use crate::load_devices;

#[derive(Tabled)]
#[tabled(rename_all = "PascalCase")]
struct CountTableEntry {
    op_name: String,
    count: usize,
}

pub fn opcount_table(op_counts: HashMap<String, usize>) -> Table {
    let mut costs = op_counts
        .iter()
        .map(|(k, v)| CountTableEntry {
            op_name: k.to_string(),
            count: *v,
        })
        .collect::<Vec<CountTableEntry>>();
    costs.sort_by(|a, b| b.count.cmp(&a.count));

    Table::new(&costs)
        .with(Style::modern())
        .with(Modify::new(Rows::first()).with(Alignment::center()))
        .with(Modify::new(Rows::new(1..)).with(Alignment::left()))
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
}

#[derive(Tabled)]
#[tabled(rename_all = "PascalCase")]
struct HardwareEntry {
    name: String,
    its: String,
}

pub fn hardware_table(total_flops: usize) -> Table {
    let devices = load_devices().unwrap();

    let hardware: Vec<HardwareEntry> = devices
        .iter()
        .map(|device| HardwareEntry {
            name: device.name,
            its: device
                .calculate_its(DType::F32, total_flops)
                .unwrap()
                .to_string(),
        })
        .collect();

    Table::new(hardware)
        .with(Style::modern())
        .with(Modify::new(Rows::first()).with(Alignment::center()))
        .with(Modify::new(Rows::new(1..)).with(Alignment::left()))
}
