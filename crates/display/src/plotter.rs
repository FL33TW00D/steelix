use std::{
    collections::{HashMap, HashSet},
    io::Write,
};

use ir::{Model, OpGroup, COLOUR_MAP};

use crate::generate_summary;

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
    dot::render(&graph, output).unwrap()
}

impl<'a> dot::Labeller<'a, Nd, Edge> for RenderableGraph {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("example2").unwrap()
    }
    fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
        dot::Id::new(format!("N{}", n)).unwrap()
    }
    fn node_label<'b>(&'b self, n: &Nd) -> dot::LabelText<'b> {
        dot::LabelText::LabelStr(self.nodes[*n].label.clone().into())
    }
    fn edge_label<'b>(&'b self, e: &Edge) -> dot::LabelText<'b> {
        dot::LabelText::LabelStr(e.label.clone().into())
    }
    fn node_color(&'a self, _node: &Nd) -> Option<dot::LabelText<'a>> {
        Some(dot::LabelText::LabelStr("black".into()))
    }

    fn node_style(&'a self, _n: &Nd) -> dot::Style {
        dot::Style::Filled
    }

    fn node_attrs(&'a self, n: &Nd) -> HashMap<&str, &str> {
        self.nodes[*n]
            .attributes
            .clone()
            .unwrap_or_else(|| HashMap::from([("fillcolor", "white")]))
    }
}

impl<'a> dot::GraphWalk<'a, Nd, Edge> for RenderableGraph {
    fn nodes(&self) -> dot::Nodes<'a, Nd> {
        (0..self.nodes.len()).collect()
    }
    fn edges(&'a self) -> dot::Edges<'a, Edge> {
        self.edges.clone().into_iter().collect()
    }
    fn source(&self, e: &Edge) -> Nd {
        e.from
    }
    fn target(&self, e: &Edge) -> Nd {
        e.to
    }
}

impl From<Model> for RenderableGraph {
    fn from(model: Model) -> Self {
        let mut g = RenderableGraph::new();
        let mut offset = 0;
        let mut op_counts = HashMap::new();
        for (op_idx, op_node) in model.nodes.iter().enumerate() {
            if op_node.op.op_group() == OpGroup::Constant {
                offset += 1;
                continue;
            }
            *op_counts.entry(op_node.name.to_owned()).or_insert(0) += 1;

            let renderable_node = g.create_node(op_node.name.clone());
            renderable_node.add_attribute((
                "fillcolor",
                COLOUR_MAP.get(&op_node.op.op_group()).unwrap_or(&"white"),
            ));
            op_node.providers.iter().for_each(|provider_id| {
                if model.nodes[*provider_id].op.op_group() != OpGroup::Constant {
                    let mut pid = *provider_id;
                    if *provider_id > offset {
                        pid -= offset;
                    }
                    g.create_edge("".to_string(), pid, op_idx - offset);
                }
            });
        }
        println!("{}", generate_summary(op_counts));
        g
    }
}
