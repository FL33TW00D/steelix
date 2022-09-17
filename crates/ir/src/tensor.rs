use std::{fmt, mem::size_of, sync::Arc};

use bytes::BytesMut;
use ndarray::{Array, ArrayViewD, ArrayViewMutD, Zip};
use num::FromPrimitive;
use onnx::onnx_pb::{self, tensor_proto::DataType as ProtoDType};

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
    pub shape: Vec<usize>,
    pub len: usize, //actual entry count
    pub data: BytesMut,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        unsafe fn eq_t<D: DataType>(cur: &Tensor, other: &Tensor) -> bool {
            cur.as_slice_unchecked::<D>() == other.as_slice_unchecked::<D>()
        }

        self.dt == other.dt
            && self.shape == other.shape
            && self.len == other.len
            && unsafe { as_float!(eq_t(self.dt)(self, other)) }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor {{\n dt: {:?}, \n shape: {:?}, \n len: {:?}, \n{}}}",
            self.dt,
            self.shape,
            self.len,
            self.stringify_data()
        )
    }
}

impl Tensor {
    pub fn new(dt: DType, shape: Vec<usize>) -> Self {
        let len = shape.iter().cloned().product::<usize>();
        let byte_count = len * dt.size_of();
        Self {
            dt,
            shape,
            len,
            data: BytesMut::with_capacity(byte_count),
        }
    }

    pub fn uninitialized<T: DataType>(shape: Vec<usize>) -> Self {
        Self::new(T::to_internal(), shape)
    }

    pub fn zeros<T: DataType>(shape: Vec<usize>) -> Self {
        let len = shape.iter().cloned().product::<usize>();
        let byte_count = len * T::to_internal().size_of();
        Self {
            dt: T::to_internal(),
            shape,
            len,
            data: BytesMut::zeroed(byte_count),
        }
    }

    pub fn arange<F: DataType + num_traits::Float>(
        shape: Vec<usize>,
        start: F,
        stop: F,
        step: F,
    ) -> Tensor {
        Array::range(start, stop, step)
            .into_shape(shape)
            .unwrap()
            .into()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().cloned().product::<usize>()
    }

    pub fn update_shape(&mut self, new_shape: Vec<usize>) {
        //todo: err check
        self.shape = new_shape;
    }

    /// Access the data as a pointer.
    pub fn as_ptr<A: DataType>(&self) -> anyhow::Result<*const A> {
        Ok(self.data.as_ptr() as *const A)
    }

    /// Access the data as a pointer.
    pub fn as_mut_ptr<A: DataType>(&mut self) -> anyhow::Result<*mut A> {
        Ok(self.data.as_mut_ptr() as *mut A)
    }

    /// Transform the data as a `ndarray::Array`.
    pub fn to_array_view<A: DataType>(&self) -> anyhow::Result<ArrayViewD<A>> {
        //TODO: error checking
        unsafe { Ok(self.to_array_view_unchecked()) }
    }

    #[inline(always)]
    pub fn get_value<A: DataType + Copy>(&self, ptr: &*const A, index: usize) -> anyhow::Result<A> {
        if index > self.len {
            anyhow::bail!("Index out of bounds: {}|{}", index, self.len);
        }
        Ok(unsafe { *ptr.add(index) })
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

    unsafe fn to_array_view_mut_unchecked<A: DataType>(&mut self) -> ArrayViewMutD<A> {
        if self.len != 0 {
            ArrayViewMutD::from_shape_ptr(&*self.shape, self.data.as_mut_ptr() as *mut A)
        } else {
            ArrayViewMutD::from_shape(&*self.shape, &mut []).unwrap()
        }
    }

    pub fn permute_axes(self, axes: &[usize]) -> Tensor {
        #[inline]
        unsafe fn permute<T: DataType>(axes: &[usize], input: Tensor) -> Tensor {
            input
                .to_array_view_unchecked::<T>()
                .permuted_axes(axes)
                .to_owned() //I don't like this to_owned here
                .into()
        }
        unsafe { as_float!(permute(self.dt)(axes, self)) }
    }

    pub fn move_axis(self, from: usize, to: usize) -> Tensor {
        let mut permutation: Vec<usize> = (0..4).collect();
        permutation.remove(from);
        permutation.insert(to, from);
        self.permute_axes(&permutation)
    }

    pub unsafe fn as_slice_unchecked<D: DataType>(&self) -> &[D] {
        std::slice::from_raw_parts::<D>(self.data.as_ptr() as *const D, self.len)
    }

    pub fn stringify_data(&self) -> String {
        unsafe fn pretty_print<D: DataType>(input: &Tensor) -> String {
            //TODO: write decent pretty printer here: https://docs.rs/ndarray/latest/src/ndarray/arrayformat.rs.html#196-199
            input.as_slice_unchecked::<D>()[0..input.len]
                .iter()
                .take(1024)
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
                .collect::<String>()
        }
        unsafe { as_float!(pretty_print(self.dt)(self)) }
    }

    pub fn all_close(self, b: &Tensor, tol: f32) -> bool {
        fn all_close_t<D>(a: Tensor, b: &Tensor, tol: D) -> bool
        where
            D: DataType + num_traits::float::Float,
        {
            let a = a.to_array_view::<D>().unwrap();
            let b = b.to_array_view::<D>().unwrap();
            Zip::from(a).and(b).all(move |a, b| {
                (a.is_nan() && b.is_nan())
                    || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
                    || (*a - *b).abs() <= tol
            })
        }

        as_float!(all_close_t(self.dt)(
            self,
            b,
            FromPrimitive::from_f32(tol).unwrap()
        ))
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
        let shape: Vec<usize> = tproto.dims.iter().map(|&i| i as usize).collect();
        let len = shape.iter().cloned().product::<usize>();
        let bytes: BytesMut = (*tproto.raw_data).into();

        Ok(Tensor {
            dt,
            shape,
            len,
            data: bytes,
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
            shape,
            len,
            data: data.into(),
        }
    }
}
