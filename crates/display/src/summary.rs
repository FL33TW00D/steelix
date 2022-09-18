use std::collections::HashMap;

use tabled::{object::Rows, Alignment, Modify, Style, Table, Tabled};

#[derive(Tabled)]
#[tabled(rename_all = "PascalCase")]
struct CostSummary {
    op_name: String,
    count: usize,
    #[tabled(rename = "Total #FLOPS")]
    flops: usize,
    #[tabled(rename = "Total #Parameters")]
    params: usize,
}

pub fn opcount_table(op_counts: HashMap<String, usize>) -> Table {
    let mut costs = op_counts
        .iter()
        .map(|(k, v)| CostSummary {
            op_name: k.to_string(),
            count: *v,
            flops: 0,
            params: 0,
        })
        .collect::<Vec<CostSummary>>();
    costs.sort_by(|a, b| b.count.cmp(&a.count));

    Table::new(&costs)
        .with(Style::modern())
        .with(Modify::new(Rows::first()).with(Alignment::center()))
        .with(Modify::new(Rows::new(1..)).with(Alignment::left()))
}
