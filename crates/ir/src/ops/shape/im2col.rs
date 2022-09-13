use ndarray::{Array, Array2, Ix2};

use crate::{DataType, Tensor};

///Im2col algorithm: Expects padded input
pub fn im2col<T: DataType + ndarray::LinalgScalar>(
    input: &Tensor,
    kernel_shape: &[usize],
    strides: (usize, usize),
    output_dims: (usize, usize),
) -> Array2<T> {
    let (kr, kc) = (kernel_shape[2], kernel_shape[3]);
    let (s0, s1) = strides;
    let (h_out, w_out) = output_dims;

    let input_shape = &input.shape;
    let (n, c, ih, iw) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );

    //TODO: do this without transpose
    let mut output = Tensor::zeros::<T>(vec![n * h_out as usize * w_out as usize, c * kr * kc]);
    let iptr = input.as_ptr::<T>().unwrap();
    let optr = output.as_mut_ptr::<T>().unwrap();
    let mut oidx = 0;
    for b in 0..n {
        let batch_offset = b * c * ih * iw;
        for h in (0..(ih - kr + 1)).step_by(s0) {
            for w in (0..(iw - kc + 1)).step_by(s1) {
                let anchor = batch_offset + (h * ih) + w;
                for channel in 0..c {
                    let channel_offset = channel * ih * iw;
                    for kri in 0..kr {
                        for kci in 0..kc {
                            unsafe {
                                *optr.add(oidx) =
                                    *iptr.add(anchor + channel_offset + (kri * iw) + kci);
                            }
                            oidx += 1;
                        }
                    }
                }
            }
        }
    }
    let output_nd = output.to_array_view::<T>().unwrap();
    let output_transpose = output_nd.t();
    let mut transposed = Array::zeros(output_transpose.raw_dim());
    transposed.assign(&output_transpose);
    transposed.into_dimensionality::<Ix2>().unwrap()
}
