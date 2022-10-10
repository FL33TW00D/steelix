use half::f16;
use std::{fmt, mem::size_of, sync::Arc};

use bytes::BytesMut;
use ndarray::{Array, ArrayViewD, ArrayViewMutD};
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
          DType::F16  => $($path)::*::<i16>($($args),*),
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
    pub data: BytesMut,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor {{\n dt: {:?}, \n shape: {:?}, \n len: {:?}, \n data: {} \n}}",
            self.dt,
            self.shape,
            self.len,
            self.stringify_data()
        )
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        unsafe fn eq_t<D: DataType>(cur: &Tensor, other: &Tensor) -> bool {
            cur.as_slice_unchecked::<D>() == other.as_slice_unchecked::<D>()
        }

        self.dt == other.dt
            && self.shape == other.shape
            && self.len == other.len
            && unsafe { as_std!(eq_t(self.dt)(self, other)) }
    }
}

impl Tensor {
    pub fn new(dt: DType, shape: Shape) -> Self {
        let len = shape.iter().product::<usize>();
        Self {
            dt,
            shape,
            len,
            data: BytesMut::zeroed(len * dt.size_of()),
        }
    }

    pub fn zeros<T: DataType>(shape: Shape) -> Self {
        let len = shape.iter().product::<usize>();
        let byte_count = len * T::to_internal().size_of();
        Self {
            dt: T::to_internal(),
            shape,
            len,
            data: BytesMut::zeroed(byte_count),
        }
    }

    pub fn from_vec<T: DataType>(shape: Shape, data: Vec<T>) -> Self {
        let len = shape.iter().product::<usize>();
        let byte_count = len * T::to_internal().size_of();
        let mut bytes = BytesMut::with_capacity(byte_count);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_count)
        });
        Self {
            dt: T::to_internal(),
            shape,
            len,
            data: bytes,
        }
    }

    pub fn uninitialized<T: DataType>(shape: Shape) -> Self {
        Self::new(T::to_internal(), shape)
    }

    pub fn uninitialized_dt(dt: DType, shape: Shape) -> Self {
        Self::new(dt, shape)
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    pub fn update_shape(&mut self, new_shape: Shape) {
        //todo: err check
        self.shape = new_shape;
    }

    ///# Safety
    /// This shit is straight unsafe dog
    pub unsafe fn as_slice_unchecked<D: DataType>(&self) -> &[D] {
        std::slice::from_raw_parts::<D>(self.data.as_ptr() as *const D, self.len)
    }

    ///# Safety
    /// This shit is extra unsafe dog
    pub unsafe fn as_mut_slice_unchecked<D: DataType>(&mut self) -> &mut [D] {
        std::slice::from_raw_parts_mut::<D>(self.data.as_ptr() as *mut D, self.len)
    }

    pub fn as_slice<D: DataType>(&self) -> Result<&[D], OpError> {
        unsafe { Ok(self.as_slice_unchecked()) }
    }

    pub fn as_mut_slice<D: DataType>(&mut self) -> Result<&mut [D], OpError> {
        //todo check len too
        unsafe { Ok(self.as_mut_slice_unchecked()) }
    }

    /// Transform the data as a `ndarray::Array`.
    pub fn to_array_view<A: DataType>(&self) -> anyhow::Result<ArrayViewD<A>> {
        //TODO: error checking
        unsafe { Ok(self.to_array_view_unchecked()) }
    }

    /// Transform the data as a `ndarray::Array`.
    pub fn to_array_view_mut<A: DataType>(&mut self) -> anyhow::Result<ArrayViewMutD<A>> {
        //TODO: error checking
        unsafe { Ok(self.to_array_view_mut_unchecked()) }
    }

    pub unsafe fn to_array_view_unchecked<A: DataType>(&self) -> ArrayViewD<A> {
        if self.len != 0 {
            ArrayViewD::from_shape_ptr(&*self.shape, self.data.as_ptr() as *const A)
        } else {
            ArrayViewD::from_shape(&*self.shape, &[]).unwrap()
        }
    }

    pub unsafe fn to_array_view_mut_unchecked<A: DataType>(&mut self) -> ArrayViewMutD<A> {
        if self.len != 0 {
            ArrayViewMutD::from_shape_ptr(&*self.shape, self.data.as_mut_ptr() as *mut A)
        } else {
            ArrayViewMutD::from_shape(&*self.shape, &mut []).unwrap()
        }
    }

    /// Access the data as a scalar.
    pub fn to_scalar<D: DataType>(&self) -> anyhow::Result<&D> {
        if self.len == 0 {
            anyhow::bail!("to_scalar called on empty tensor ({:?})", self)
        }
        unsafe { Ok(self.to_scalar_unchecked()) }
    }

    /// Access the data as a scalar.
    pub unsafe fn to_scalar_unchecked<D: DataType>(&self) -> &D {
        &*(self.data.as_ptr() as *mut D)
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn stack_tensors(
        axis: usize,
        tensors: &[impl std::borrow::Borrow<Tensor>],
    ) -> anyhow::Result<Tensor> {
        let rank = tensors[0].borrow().rank();
        let dt = tensors[0].borrow().dt;
        let mut shape: Shape = tensors[0].borrow().shape.clone();
        for ax in 0..rank {
            if ax != axis {
                anyhow::ensure!(tensors.iter().all(|t| t.borrow().shape[ax] == shape[ax]));
            }
        }
        shape[axis] = tensors.iter().map(|v| v.borrow().shape[axis]).sum();
        unsafe {
            let mut result = Tensor::uninitialized_dt(dt, shape.clone());
            if shape[..axis].iter().all(|d| *d == 1) {
                let mut offset = 0isize;
                for v in tensors {
                    let v = v.borrow();
                    let dd = v.data.as_ptr();
                    std::ptr::copy_nonoverlapping(
                        dd,
                        result.data.as_mut_ptr().offset(offset),
                        v.len,
                    );
                    offset += v.len as isize;
                }
            }
            Ok(result)
        }
    }

    //Rust generics fucking suck or I'd extract this to a function
    /*
    pub fn stringify_data(&self) -> String {
        unsafe fn pretty_print<D: DataType>(input: &Tensor) -> String {
            let chunk_size = if inputproto.len < 64 { inputproto.len - 1 } else { 64 };
            let start_chunk = &inputproto.as_slice::<D>().unwrap()[0..chunk_size];

            let start_str = start_chunk
                .iter()
                .enumerate()
                .map(|(idx, d)| {
                    let mut out = format!("{:>10.6},", d);
                    if (inputproto.shape.len() > 1)
                        && (idx + 1).rem_euclid(inputproto.shape[inputproto.shape.len() - 1]) == 0
                    {
                        outproto.push('\n')
                    }
                    out
                })
                .collect::<String>();
            start_str
        }
        unsafe { as_std!(pretty_print(self.dt)(self)) }
    }
    */
    pub fn stringify_data(&self) -> String {
        unsafe fn pretty_print<D: DataType>(input: &Tensor) -> String {
            input
                .to_array_view::<D>()
                .expect("Failed to convert to array view")
                .to_string()
        }
        unsafe { as_std!(pretty_print(self.dt)(self)) }
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
    F16,
    #[default]
    F32,
    F64,
}

impl DType {
    #[inline]
    pub fn size_of(&self) -> usize {
        as_std!(std::mem::size_of(self)())
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
map_type!(f16, F16); //half crate
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
            ProtoDType::Float16 => Ok(DType::F16),
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

        let tensor = if !tproto.raw_data.is_empty() {
            let len = shape.iter().cloned().product::<usize>();
            let data: BytesMut = (*tproto.raw_data).into();
            Tensor {
                dt,
                shape,
                len,
                data,
            }
        } else {
            match dt {
                DType::U8 => {
                    Tensor::from_vec(shape, tproto.int32_data.iter().map(|&x| x as u8).collect())
                }
                DType::U16 => {
                    Tensor::from_vec(shape, tproto.int32_data.iter().map(|&x| x as u16).collect())
                }
                DType::U32 => Tensor::from_vec(shape, tproto.int32_data.to_vec()),
                DType::U64 => Tensor::from_vec(shape, tproto.int64_data.to_vec()),
                DType::I8 => {
                    Tensor::from_vec(shape, tproto.int32_data.iter().map(|&x| x as i8).collect())
                }
                DType::I16 => {
                    Tensor::from_vec(shape, tproto.int32_data.iter().map(|&x| x as i16).collect())
                }
                DType::I32 => Tensor::from_vec(shape, tproto.int32_data.to_vec()),
                DType::I64 => Tensor::from_vec(shape, tproto.int64_data.to_vec()),
                DType::F32 => Tensor::from_vec(shape, tproto.float_data.to_vec()),
                DType::F64 => Tensor::from_vec(shape, tproto.double_data.to_vec()),
            }
        };

        Ok(tensor)
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
            data: data.into(),
        }
    }
}
