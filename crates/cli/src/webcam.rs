use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use image::DynamicImage;
use image::ImageBuffer;
use ir::IntoArcTensor;
use ir::Model;
use ir::Tensor;
use opencv::{core::Scalar, highgui, imgproc, prelude::*, videoio::VideoCapture, videoio::CAP_ANY};

use crate::get_result_label;
use crate::preprocess_image;

pub fn infer_webcam(mut runnable: Model) {
    let window = "Steelix Demo";
    highgui::named_window(window, 1).unwrap();
    let mut cam = VideoCapture::new(0, CAP_ANY).unwrap(); // 0 is the default camera
    let opened = cam.is_opened().unwrap();
    if !opened {
        panic!("Unable to open default camera!");
    }

    let mut loopidx: u64 = 0;
    let mut label = "Starting...".to_string();
    loop {
        let order = runnable.build_traversal_order();
        let mut frame = Mat::default();
        cam.read(&mut frame).unwrap();
        let frame_bytes: Vec<u8> = frame.data_bytes().unwrap().into();
        let webcam_image =
            DynamicImage::ImageRgb8(ImageBuffer::from_raw(1280, 720, frame_bytes).unwrap())
                .to_rgb8();
        if loopidx % 10 == 0 {
            let image_tensor = preprocess_image(webcam_image);
            let inputs: HashMap<String, Arc<Tensor>> =
                HashMap::from([("images:0".into(), image_tensor.into_arc_tensor())]);
            let start = Instant::now();
            let run_result = runnable.run(inputs, order).unwrap();
            println!("Time elapsed: {:?}", start.elapsed());
            label = get_result_label(run_result);
        }
        imgproc::put_text(
            &mut frame,
            &label,
            opencv::core::Point_ { x: 100, y: 100 },
            imgproc::FONT_HERSHEY_SIMPLEX,
            1.5,
            Scalar::from((255., 255., 255.)),
            4,
            imgproc::LINE_8,
            false,
        )
        .unwrap();
        highgui::imshow(window, &frame).unwrap();
        highgui::wait_key(1).unwrap();
        loopidx += 1;
    }
}
