use std::{fmt, mem::size_of, sync::Arc};

use bytes::BytesMut;
use ndarray::Array;
use onnx::onnx_pb::{self, tensor_proto::DataType as ProtoDType};

use crate::{OpError, Shape};

#[macro_export]
macro_rules! as_std {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        match $dt {
          DType::U8   => $($path)::*::<u8>($($args),*),
          DType::U16  => $($path)::*::<u16>($($args),*),
          DType::U32  => $($path)::*::<u32>($($args),*),
          DType::U64  => $($path)::*::<u64>($($args),*),
          DType::I8   => $($path)::*::<i8>($($args),*),
          DType::I16  => $($path)::*::<i16>($($args),*),
          DType::I32  => $($path)::*::<i32>($($args),*),
          DType::I64  => $($path)::*::<i64>($($args),*),
          DType::F32  => $($path)::*::<f32>($($args),*),
          DType::F64  => $($path)::*::<f64>($($args),*),
        }
    } }
}

#[macro_export]
macro_rules! as_float {
    ($($path:ident)::* ($dt:expr) ($($args:expr),*)) => { {
        match $dt {
          DType::F32  => $($path)::*::<f32>($($args),*),
          DType::F64  => $($path)::*::<f64>($($args),*),
          _ => panic!("Called float op with incorrect dtype")
        }
    } }
}
#[derive(Clone, Default)]
pub struct Tensor {
    pub dt: DType,
    pub shape: Shape,
    pub len: usize, //actual entry count
    pub data: Option<BytesMut>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor {{\n dt: {:?}, \n shape: {:?}, \n len: {:?}, \n {}}}",
            self.dt,
            self.shape,
            self.len,
            self.stringify_data()
        )
    }
}

impl Tensor {
    pub fn new(dt: DType, shape: Shape, data: Option<BytesMut>) -> Self {
        let len = shape.iter().cloned().product::<usize>();
        Self {
            dt,
            shape,
            len,
            data,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().cloned().product::<usize>()
    }

    pub fn update_shape(&mut self, new_shape: Shape) {
        //todo: err check
        self.shape = new_shape;
    }

    ///# Safety
    /// This shit is straight unsafe dog
    pub unsafe fn as_slice_unchecked<D: DataType>(&self) -> &[D] {
        std::slice::from_raw_parts::<D>(self.data.as_ref().unwrap().as_ptr() as *const D, self.len)
    }

    pub fn as_slice<D: DataType>(&self) -> Result<&[D], OpError> {
        if self.data.is_none() {
            Err(OpError::ValidationError(
                "Tried to take slice of non existent data!".to_string(),
            ))
        } else {
            //todo check len too
            unsafe { Ok(self.as_slice_unchecked()) }
        }
    }

    //Rust generics fucking suck or I'd extract this to a function
    pub fn stringify_data(&self) -> String {
        unsafe fn pretty_print<D: DataType>(input: &Tensor) -> String {
            let chunk_size = 64;
            let start_chunk = &input.as_slice::<D>().unwrap()[0..chunk_size];
            let end_chunk = &input.as_slice::<D>().unwrap()[input.len - chunk_size..input.len];

            let start_str = start_chunk
                .iter()
                .enumerate()
                .map(|(idx, d)| {
                    let mut out = format!("{:>10.6},", d);
                    if (input.shape.len() > 1)
                        && (idx + 1).rem_euclid(input.shape[input.shape.len() - 1]) == 0
                    {
                        out.push('\n')
                    }
                    out
                })
                .collect::<String>();
            let end_str = end_chunk
                .iter()
                .enumerate()
                .map(|(idx, d)| {
                    let mut out = format!("{:>10.6},", d);
                    if (input.shape.len() > 1)
                        && (idx + 1).rem_euclid(input.shape[input.shape.len() - 1]) == 0
                    {
                        out.push('\n')
                    }
                    out
                })
                .collect::<String>();

            format!("{}\n...\n{}", start_str, end_str)
        }
        unsafe { as_float!(pretty_print(self.dt)(self)) }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Default)]
pub enum DType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    #[default]
    F32,
    F64,
}

impl DType {
    #[inline]
    pub fn size_of(&self) -> usize {
        as_float!(std::mem::size_of(self)())
    }
}

///DataType trait is implemented for all supported std types
pub trait DataType: Clone + fmt::Debug + fmt::Display + PartialEq {
    fn to_internal() -> DType;
}

macro_rules! map_type {
    ($t:ty, $v:ident) => {
        impl DataType for $t {
            fn to_internal() -> DType {
                DType::$v
            }
        }
    };
}

map_type!(u8, U8);
map_type!(u16, U16);
map_type!(u32, U32);
map_type!(u64, U64);
map_type!(i8, I8);
map_type!(i16, I16);
map_type!(i32, I32);
map_type!(i64, I64);
map_type!(f32, F32);
map_type!(f64, F64);

impl TryFrom<ProtoDType> for DType {
    type Error = anyhow::Error;

    fn try_from(proto_dt: ProtoDType) -> Result<Self, Self::Error> {
        match proto_dt {
            ProtoDType::Int8 => Ok(DType::I8),
            ProtoDType::Undefined => todo!(),
            ProtoDType::Float => Ok(DType::F32),
            ProtoDType::Uint8 => Ok(DType::U8),
            ProtoDType::Uint16 => Ok(DType::U16),
            ProtoDType::Int16 => Ok(DType::I16),
            ProtoDType::Int32 => Ok(DType::I32),
            ProtoDType::Int64 => Ok(DType::I64),
            ProtoDType::String => todo!(),
            ProtoDType::Bool => todo!(),
            ProtoDType::Float16 => todo!(),
            ProtoDType::Double => Ok(DType::F32),
            ProtoDType::Uint32 => Ok(DType::U32),
            ProtoDType::Uint64 => Ok(DType::U64),
            ProtoDType::Complex64 => todo!(),
            ProtoDType::Complex128 => todo!(),
            ProtoDType::Bfloat16 => todo!(),
        }
    }
}

impl TryFrom<onnx_pb::TensorProto> for Tensor {
    type Error = anyhow::Error;

    fn try_from(tproto: onnx_pb::TensorProto) -> Result<Self, Self::Error> {
        let dt = ProtoDType::from_i32(tproto.data_type).unwrap().try_into()?;
        let shape: Shape = tproto.dims.iter().map(|&i| i as usize).collect();
        let len = shape.iter().cloned().product::<usize>();
        let data: BytesMut = (*tproto.raw_data).into();

        Ok(Tensor {
            dt,
            shape,
            len,
            data: Some(data),
        })
    }
}

/// Convenient conversion to Tensor.
pub trait IntoTensor: Sized {
    /// Convert Self to a Tensor.
    ///
    /// May perform a copy
    fn into_tensor(self) -> Tensor;
}

/// Convenient conversion to Arc<Tensor>.
pub trait IntoArcTensor: Sized {
    /// Convert Self to a Arc<Tensor>.
    ///
    /// May perform a copy
    fn into_arc_tensor(self) -> Arc<Tensor>;
}

impl<D: ::ndarray::Dimension, T: DataType> IntoArcTensor for Array<T, D> {
    fn into_arc_tensor(self) -> Arc<Tensor> {
        Arc::new(Tensor::from(self))
    }
}

impl IntoTensor for Arc<Tensor> {
    fn into_tensor(self) -> Tensor {
        Arc::try_unwrap(self).unwrap_or_else(|t| (*t).clone())
    }
}

impl IntoArcTensor for Tensor {
    fn into_arc_tensor(self) -> Arc<Tensor> {
        Arc::new(self)
    }
}

impl IntoArcTensor for Arc<Tensor> {
    fn into_arc_tensor(self) -> Arc<Tensor> {
        self
    }
}

impl<D: ::ndarray::Dimension, T: DataType> IntoTensor for Array<T, D> {
    fn into_tensor(self) -> Tensor {
        Tensor::from(self)
    }
}

//A is elem type, D is dimension
impl<A: DataType, D: ::ndarray::Dimension> From<Array<A, D>> for Tensor {
    fn from(nda: Array<A, D>) -> Tensor {
        let shape = nda.shape().to_vec();
        let vec = nda.into_raw_vec().into_boxed_slice();
        let len = vec.len();
        let byte_count = len * size_of::<A>();
        let data = unsafe { std::slice::from_raw_parts(Box::into_raw(vec) as *mut u8, byte_count) };
        Tensor {
            dt: A::to_internal(),
            shape: shape.into(),
            len,
            data: Some(data.into()),
        }
    }
}
