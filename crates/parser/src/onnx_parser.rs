use ir::{ops::data::Unimplemented, *};
use onnx::*;
use prost::Message;
use std::collections::{HashMap, HashSet};

///Parses a valid ONNX model at the provided path
pub fn parse_model(model_path: &std::path::PathBuf) -> Result<Model, anyhow::Error> {
    let pb_model = onnx_pb::ModelProto::decode(bytes::Bytes::from(std::fs::read(model_path)?))?;
    let pb_graph = pb_model.graph.expect("No model graph found.");

    let mut model = Model::new();

    let mut initializers_map = parse_graph_initializers(&pb_graph.initializer);
    let inputs_map = parse_graph_inputs(&pb_graph.input, &mut initializers_map, &mut model);

    let initializer_ids =
        initializers_map
            .into_iter()
            .fold(HashMap::new(), |mut acc, (name, tensor)| {
                acc.insert(
                    name.clone(),
                    model.add_node(name, ops::data::build_constant(tensor).unwrap()),
                );
                acc
            });

    create_graph_nodes(&mut model, &pb_graph.node, OpRegister::default());

    let outputs_map = parse_graph_outputs(&pb_graph.output, &mut model);
    link_nodes(
        &mut model,
        &pb_graph,
        inputs_map,
        outputs_map,
        initializer_ids,
    );
    Ok(model)
}

///Model initializers from ONNX file
fn parse_graph_initializers(initializers: &[onnx_pb::TensorProto]) -> HashMap<String, Tensor> {
    initializers.iter().fold(HashMap::new(), |mut acc, ip| {
        acc.insert(ip.name.to_owned(), ip.clone().try_into().unwrap());
        acc
    })
}

///User provided graph inputs
fn parse_graph_inputs(
    inputs: &[onnx_pb::ValueInfoProto],
    initializers_map: &mut HashMap<String, Tensor>,
    model: &mut Model,
) -> HashMap<String, usize> {
    let mut inputs_map = HashMap::new();
    for (input_idx, input) in inputs.iter().enumerate() {
        if let Some(init) = initializers_map.remove(&*input.name) {
            model.add_node(
                input.name.to_owned(),
                ops::data::build_constant(init).unwrap(), //static constants
            );
        } else {
            println!("INPUTS: {:?}", inputs);
            let input_node_id = model.add_node(
                input.name.to_owned(),
                ops::data::build_initial((*input).clone().try_into().unwrap()).unwrap(),
            );
            model.inputs.push(input_node_id);
            inputs_map.insert(input.name.to_owned(), input_idx);
        }
    }
    inputs_map
}

fn parse_graph_outputs(
    outputs: &[onnx_pb::ValueInfoProto],
    model: &mut Model,
) -> HashMap<String, usize> {
    let mut outputs_map = HashMap::new();
    let offset = model.nodes.len();
    for (output_idx, output) in outputs.iter().enumerate() {
        let output_node_id =
            model.add_node(output.name.to_owned(), Box::new(Unimplemented) as BoxOp);
        model.outputs.push(output_node_id);
        outputs_map.insert(output.name.to_owned(), offset + output_idx);
    }
    outputs_map
}

fn link_nodes(
    model: &mut Model,
    model_graph: &onnx_pb::GraphProto,
    inputs_map: HashMap<String, usize>,
    outputs_map: HashMap<String, usize>,
    initializers_map: HashMap<String, usize>,
) {
    let graph_nodes = &model_graph.node;

    let graph_offset = model.inputs.len() + initializers_map.len();
    for (op_idx, op_node) in graph_nodes.iter().enumerate() {
        op_node.input.iter().for_each(|input| {
            if inputs_map.contains_key(input) {
                model.add_edge(*inputs_map.get(input).unwrap(), op_idx + graph_offset);
            }

            if initializers_map.contains_key(input) {
                model.add_edge(*initializers_map.get(input).unwrap(), op_idx + graph_offset);
            }
        });

        op_node.output.iter().for_each(|output| {
            if outputs_map.contains_key(output) {
                model.add_edge(op_idx + graph_offset, *outputs_map.get(output).unwrap());
            }
        });

        let output_set: HashSet<String> = HashSet::from_iter(op_node.output.iter().cloned());
        let target_offset = op_idx + 1; //topo offset
        for (target_idx, target_node) in graph_nodes[target_offset..graph_nodes.len()]
            .iter()
            .enumerate()
        {
            let target_inp_set: HashSet<String> =
                HashSet::from_iter(target_node.input.iter().cloned());

            if !target_inp_set.is_disjoint(&output_set) {
                let producer_id = op_idx + graph_offset;
                let consumer_id = target_idx + target_offset + graph_offset;
                //println!("INSERTING EDGE: {:?} -> {:?}", producer_id, consumer_id);
                model.add_edge(producer_id, consumer_id);
            }
        }
    }
}

fn create_graph_nodes(
    model: &mut Model,
    graph_nodes: &[onnx_pb::NodeProto],
    op_register: OpRegister,
) {
    for op_node in graph_nodes.iter() {
        let op = match op_register.get(&op_node.op_type.clone()) {
            Some(builder) => (builder)(op_node).unwrap(),
            None => ops::data::build_unimplemented(op_node).unwrap(),
        };

        model.add_node(op_node.op_type.clone(), op);
    }
}
