use npyz::WriterBuilder;
use std::{fs::File, io};

use crate::Tensor;

fn write_array<T, S, D>(writer: impl io::Write, array: &ndarray::ArrayBase<S, D>) -> io::Result<()>
where
    T: Clone + npyz::AutoSerialize,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
{
    let shape = array.shape().iter().map(|&x| x as u64).collect::<Vec<_>>();
    let c_order_items = array.iter();

    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&shape)
        .writer(writer)
        .begin_nd()?;
    writer.extend(c_order_items)?;
    writer.finish()
}

pub fn dump_tensor(fname: &str, input: &Tensor) {
    let mut file = io::BufWriter::new(File::create(fname).unwrap());
    let d = &input.to_array_view::<f32>().unwrap();
    write_array(&mut file, d).unwrap();
}
