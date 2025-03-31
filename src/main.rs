use std::time::Instant;
use crate::perspective::Equirectangular;
use opencv;

mod perspective;

fn main() {
    let start = Instant::now();

    let image = Equirectangular::new("image.jpg");
    let perspective_image = image.get_perspective(60.0, 80.0, 33.0, 720, 1080);
    opencv::imgcodecs::imwrite("final_image.jpg", &perspective_image, &opencv::core::Vector::<i32>::new()).expect("Could not write image!");

    println!("{}", start.elapsed().as_secs_f64());
}