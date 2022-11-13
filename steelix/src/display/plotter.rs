use std::{collections::HashMap, io::Write};

use crate::{
    ir::{Model, ModelSummary, OpGroup, COLOUR_MAP, SHAPE_MAP},
    opcount_table,
};

type Nd = usize;

#[derive(Default, Debug)]
pub struct RenderableGraph {
    pub current_id: usize,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

#[derive(Clone, Debug)]
pub struct Edge {
    label: String, //COW
    from: usize,
    to: usize,
}

impl Edge {
    pub fn new(label: String, from: usize, to: usize) -> Self {
        Self { label, from, to }
    }
}

impl RenderableGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_node(&mut self, label: String) -> &mut Node {
        let n = Node {
            id: self.current_id,
            label,
            attributes: None,
        };
        self.nodes.push(n);
        self.current_id += 1;
        &mut self.nodes[self.current_id - 1]
    }

    pub fn create_edge(&mut self, label: String, from: usize, to: usize) {
        self.edges.push(Edge::new(label, from, to));
    }

    pub fn build_graph(model: Model, model_summary: Option<ModelSummary>) -> Self {
        let mut g = RenderableGraph::new();
        let mut offset = 0;

        let mut op_counts = HashMap::new();
        for (op_idx, op_node) in model.nodes.iter().enumerate() {
            op_counts
                .entry(op_node.op.name().to_string().clone())
                .and_modify(|e| *e += 1)
                .or_insert(1);
            if op_node.op.op_group() == OpGroup::Constant {
                offset += 1;
                continue;
            }

            let renderable_node = g.create_node(op_node.name.clone());
            //TODO: move below to function
            renderable_node.add_attribute((
                "fillcolor",
                COLOUR_MAP.get(&op_node.op.op_group()).unwrap_or(&"white"),
            ));
            renderable_node.add_attribute((
                "shape",
                SHAPE_MAP.get(&op_node.op.op_group()).unwrap_or(&"ellipse"),
            ));
            op_node.providers.iter().for_each(|provider_id| {
                if model.nodes[*provider_id].op.op_group() != OpGroup::Constant {
                    let mut pid = *provider_id;
                    let shape = if let Some(summary) = &model_summary {
                        summary.output_shapes.get(&pid).unwrap().to_string()
                    } else {
                        "".to_string()
                    };
                    if *provider_id > offset {
                        pid -= offset;
                    }
                    g.create_edge(shape, pid, op_idx - offset);
                }
            });
        }
        let opcount_table = opcount_table(op_counts);
        println!("{}", opcount_table);
        g
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub label: String,
    pub attributes: Option<HashMap<&'static str, &'static str>>,
}

impl Node {
    pub fn add_attribute(&mut self, attribute: (&'static str, &'static str)) {
        if let Some(attr_map) = &mut self.attributes {
            attr_map.insert(attribute.0, attribute.1);
        } else {
            self.attributes = Some(HashMap::from([attribute]));
        }
    }
}

pub fn render_to<W: Write>(output: &mut W, graph: RenderableGraph) {
    dot3::render(&graph, output).unwrap()
}

impl<'a> dot3::Labeller<'a, Nd, Edge> for RenderableGraph {
    fn graph_id(&'a self) -> dot3::Id<'a> {
        dot3::Id::new("example2").unwrap()
    }
    fn node_id(&'a self, n: &Nd) -> dot3::Id<'a> {
        dot3::Id::new(format!("N{}", n)).unwrap()
    }
    fn node_label<'b>(&'b self, n: &Nd) -> dot3::LabelText<'b> {
        dot3::LabelText::LabelStr(self.nodes[*n].label.clone().into())
    }
    fn edge_label<'b>(&'b self, e: &Edge) -> dot3::LabelText<'b> {
        dot3::LabelText::LabelStr(e.label.clone().into())
    }
    fn node_color(&'a self, _node: &Nd) -> Option<dot3::LabelText<'a>> {
        Some(dot3::LabelText::LabelStr("black".into()))
    }

    fn node_style(&'a self, _n: &Nd) -> dot3::Style {
        dot3::Style::Filled
    }

    fn node_attrs(&'a self, n: &Nd) -> HashMap<&str, &str> {
        self.nodes[*n]
            .attributes
            .clone()
            .unwrap_or_else(|| HashMap::from([("fillcolor", "white")]))
    }
}

impl<'a> dot3::GraphWalk<'a, Nd, Edge> for RenderableGraph {
    fn nodes(&self) -> dot3::Nodes<'a, Nd> {
        (0..self.nodes.len()).collect()
    }
    fn edges(&'a self) -> dot3::Edges<'a, Edge> {
        self.edges.clone().into_iter().collect()
    }
    fn source(&self, e: &Edge) -> Nd {
        e.from
    }
    fn target(&self, e: &Edge) -> Nd {
        e.to
    }
}
