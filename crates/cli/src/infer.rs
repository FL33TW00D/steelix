use image::{ImageBuffer, Rgb};
use ir::{IntoArcTensor, Model, Tensor};
use ndarray::Array4;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use std::time::Instant;

pub fn get_result_label(run_result: Arc<Tensor>) -> String {
    let max_idx = get_argmax(&run_result);
    let file = File::open("./resources/models/eff-lite/eff_labels_map.json").unwrap();
    let reader = BufReader::new(file);

    let u: serde_json::Value = serde_json::from_reader(reader).unwrap();
    let result = &u[max_idx.to_string()];
    println!("Prediction: {:?}", result);
    result.to_string()
}

use ordered_float::NotNan;
fn get_argmax(result: &Tensor) -> usize {
    unsafe {
        let non_nan_floats: Vec<NotNan<f32>> = result
            .as_slice_unchecked()
            .iter()
            .cloned()
            .map(NotNan::new) // Attempt to convert each f32 to a NotNan
            .filter_map(Result::ok) // Unwrap the `NotNan`s and filter out the `NaN` values
            .collect();
        let max = non_nan_floats.iter().max().unwrap();
        non_nan_floats
            .iter()
            .position(|element| element == max)
            .unwrap()
    }
}

pub fn preprocess_image(image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Tensor {
    let resized = image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Nearest);
    Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    })
    .into()
}

pub fn infer_path(mut runnable: Model, image_path: String) {
    let image = image::open::<String>(image_path).unwrap().to_rgb8();
    let image_tensor = preprocess_image(image);

    let inputs: HashMap<String, Arc<Tensor>> =
        HashMap::from([("images:0".into(), image_tensor.into_arc_tensor())]);

    let order = runnable.build_traversal_order();
    let start = Instant::now();
    let run_result = runnable.run(inputs, order).unwrap();
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    get_result_label(run_result);
}
