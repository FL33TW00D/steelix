use crate::{shape, Shape};

pub fn multi_broadcast(shapes: &[Shape]) -> Option<Shape> {
    let len = shapes.iter().map(|shape| shape.as_ref().len()).max()?;
    let mut shape: Shape = shape!();
    for i in 0..len {
        let mut wanted_size = 1;
        for shape in shapes {
            let len = shape.as_ref().len();
            let dim = if i < len {
                &shape.as_ref()[len - i - 1]
            } else {
                &1
            };
            if dim != &1 {
                if wanted_size != 1 && dim != &wanted_size {
                    return None;
                }
                wanted_size = *dim;
            }
        }
        shape.push(wanted_size)
    }
    shape.reverse();
    Some(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_1() {
        assert_eq!(
            multi_broadcast(&[shape![2, 3, 4, 5], shape![]]),
            Some(shape![2, 3, 4, 5])
        )
    }

    #[test]
    fn onnx_2() {
        assert_eq!(
            multi_broadcast(&[shape![2, 3, 4, 5], shape![5]]),
            Some(shape![2, 3, 4, 5])
        )
    }

    #[test]
    fn onnx_3() {
        assert_eq!(
            multi_broadcast(&[shape![4, 5], shape![2, 3, 4, 5]]),
            Some(shape![2, 3, 4, 5])
        )
    }

    #[test]
    fn onnx_4() {
        assert_eq!(
            multi_broadcast(&[shape![1, 4, 5], shape![2, 3, 4, 1]]),
            Some(shape![2, 3, 4, 5])
        )
    }

    #[test]
    fn onnx_5() {
        assert_eq!(
            multi_broadcast(&[shape![3, 4, 5], shape![2, 1, 1, 1]]),
            Some(shape![2, 3, 4, 5])
        )
    }
}
