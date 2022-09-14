use crate::{BoxOp, Op, OpNode, Tensor};

impl<T: Op + ?Sized> Op for Box<T> {
    #[inline]
    fn name(&self) -> std::borrow::Cow<str> {
        (**self).name()
    }

    #[inline]
    fn op_group(&self) -> crate::OpGroup {
        (**self).op_group()
    }
}

use core::fmt::Debug;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
impl Debug for dyn Op {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error("{0}")]
    ValidationError(String),
    #[error(transparent)]
    UnexpectedError(#[from] anyhow::Error),
}

///                         MODEL NODE STRUCTURE
/// --------------------------------------------------------------------
/// | user inputs | constant initializers | processing nodes | outputs |
/// --------------------------------------------------------------------
#[derive(Debug, Default)]
pub struct Model {
    pub nodes: Vec<OpNode<BoxOp>>,
    pub inputs: Vec<usize>,  //IDs of input nodes
    pub outputs: Vec<usize>, //IDs of output nodes
}

pub struct TraversalState {
    pub intermediates: HashMap<usize, Arc<Tensor>>,
    //eviction_state
}

impl Model {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, name: String, op: BoxOp) -> usize {
        let id = self.nodes.len();
        self.nodes.push(OpNode {
            id,
            name,
            op,
            providers: vec![],
            consumers: vec![],
        });
        id
    }

    pub fn add_edge(&mut self, producer_id: usize, consumer_id: usize) {
        let producer = &mut self.nodes[producer_id];
        producer.consumers.push(consumer_id);

        let consumer = &mut self.nodes[consumer_id];
        consumer.providers.push(producer_id);
    }

    ///Performs a DFS from each target node
    pub fn build_traversal_order(&self) -> Vec<usize> {
        let mut visited = HashSet::with_capacity(self.nodes.len());
        let mut order: Vec<usize> = vec![];
        for target in self.outputs.clone() {
            if visited.contains(&target) {
                continue;
            }
            let mut current_stack: Vec<(usize, usize)> = vec![(target, 0)]; //node_idx, provider_idx
            while let Some((current_node, current_input)) = current_stack.pop() {
                if self.inputs.contains(&current_node) //reached leaf or all providers visited
                    || current_input == self.nodes[current_node].providers.len()
                {
                    order.push(current_node);
                    visited.insert(current_node);
                } else {
                    let provider = self.nodes[current_node].providers[current_input];

                    if visited.contains(&provider) {
                        //If provider has been added to order, then we know we can move on to
                        //next provider
                        current_stack.push((current_node, current_input + 1));
                    } else {
                        //If we haven't seen the provider, readd the current node
                        //to the stack, and add the provider node to the stack
                        current_stack.push((current_node, current_input));
                        current_stack.push((provider, 0));
                    }
                }
            }
        }
        order
    }

    pub fn traverse(
        &mut self,
        initials: HashMap<String, Arc<Tensor>>,
        mut order: Vec<usize>,
    ) -> Result<Arc<Tensor>, ModelError> {
        order.pop(); //remove the final node
        let mut traversal_state = TraversalState {
            intermediates: HashMap::new(),
        };

        for input_id in &self.inputs {
            let input_node = &mut self.nodes[*input_id];
            let input_initial = initials.get(&input_node.name).ok_or_else(|| {
                ModelError::ValidationError("Failed to get required input.".to_string())
            })?;

            //(*input_node.op).update(Arc::clone(input_initial));
        }

        for node_id in order {
            let node = &mut self.nodes[node_id];

            let providers: Vec<Arc<Tensor>> = node
                .providers
                .iter()
                .map(|id| Arc::clone(traversal_state.intermediates.get(id).unwrap()))
                .collect();

            let result = vec![Arc::new(Tensor::default())];
            traversal_state
                .intermediates
                .insert(node_id, Arc::clone(&result[0]));
        }
        let result = traversal_state
            .intermediates
            .get(&(self.outputs[0] - 1))
            .unwrap()
            .clone();
        Ok(result)
    }

    pub fn run(
        &mut self,
        initials: HashMap<String, Arc<Tensor>>,
        order: Vec<usize>,
    ) -> Result<Arc<Tensor>, ModelError> {
        Self::traverse(self, initials, order)
    }
}
