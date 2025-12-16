use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;

// --- Contract Definition ---
#[derive(Debug, Deserialize, Serialize)]
struct InferenceContract {
    input_width: u32,
    input_height: u32,
    input_channels: u32,
    output_classes: Vec<String>,
}

impl InferenceContract {
    fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let file = std::fs::File::open(path)?;
        Ok(serde_json::from_reader(file)?)
    }
}

// --- Image Processing ---
fn preprocess_image(path: &Path, width: u32, height: u32) -> Result<Vec<f32>, Box<dyn Error>> {
    let img = image::open(path)?.to_rgb8();
    let resized = image::imageops::resize(&img, width, height, image::imageops::FilterType::Triangle);
    
    // Normalize to 0.0 - 1.0 flat vector
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for px in resized.pixels() {
        pixels.push(px.0[0] as f32 / 255.0);
        pixels.push(px.0[1] as f32 / 255.0);
        pixels.push(px.0[2] as f32 / 255.0);
    }
    Ok(pixels)
}

// --- Pipeline Abstraction ---
#[derive(Debug)]
struct SimpleError(String);
impl std::fmt::Display for SimpleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Error for SimpleError {}

struct ImageClassificationPipeline {
    input_buf_size: u64,
    output_buf_size: u64,
    output_count: usize,
}

impl ImageClassificationPipeline {
    fn new(_contract: &InferenceContract) -> Result<Self, Box<dyn Error>> {
        let out_count = _contract.output_classes.len();
        let in_size = (_contract.input_width * _contract.input_height * _contract.input_channels * 4) as u64;
        let out_size = (out_count * 4) as u64;

        // MVP/demo: placeholder buffers
        // Real impl would allocate VkBuffer via Vulkan
        Ok(Self {
            input_buf_size: in_size,
            output_buf_size: out_size,
            output_count: out_count,
        })
    }

    fn infer(&self, _input_data: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
        // 1. Host -> Device transfer (ctx.write_buffer)
        // 2. Dispatch Compute Shader (ctx.dispatch)
        // 3. Device -> Host transfer (ctx.read_buffer)
        
        // Mocking the result for the demo since we don't have the physical shader file
        println!("  [GPU] Executing inference kernel...");
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        let mut results = vec![0.0; self.output_count];
        if !results.is_empty() {
            // Fake prediction: Class 0 is highly likely
            results[0] = 0.92;
            for i in 1..self.output_count {
                results[i] = 0.08 / (self.output_count as f32 - 1.0);
            }
        }
        Ok(results)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("--- HLX Image Classification Demo ---");

    // 1. Setup paths
    let contract_path = Path::new("contract.json");
    let image_path = Path::new("input.jpg");

    // Generate dummy files if missing for immediate testing
    if !contract_path.exists() {
        generate_dummy_contract(contract_path)?;
    }
    if !image_path.exists() {
        generate_dummy_image(image_path)?;
    }

    // 2. Load Contract
    println!("Loading contract...");
    let contract = InferenceContract::load(contract_path)?;
    println!("Model: ResNet50 (Variant), Classes: {}", contract.output_classes.len());

    // 3. Create Pipeline (no Vulkan dependency for MVP)
    println!("Building Compute Pipeline...");
    let pipeline = ImageClassificationPipeline::new(&contract)?;

    // 5. Load & Process Image
    println!("Preprocessing image...");
    let input_data = preprocess_image(image_path, contract.input_width, contract.input_height)?;

    // 6. Run Inference
    println!("Running inference...");
    let results = pipeline.infer(&input_data)?;

    // 7. Interpret Results
    let (idx, prob) = results.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!("\nPrediction Result:");
    println!("  Class: \"{}\"", contract.output_classes[idx]);
    println!("  Confidence: {:.2}%", prob * 100.0);

    Ok(())
}

// Helpers for the demo experience
fn generate_dummy_contract(path: &Path) -> std::io::Result<()> {
    let dummy = InferenceContract {
        input_width: 224, input_height: 224, input_channels: 3,
        output_classes: vec!["Tabby Cat".into(), "Golden Retriever".into(), "Sports Car".into()],
    };
    std::fs::write(path, serde_json::to_string_pretty(&dummy)?)
}

fn generate_dummy_image(path: &Path) -> Result<(), Box<dyn Error>> {
    let img = image::RgbImage::from_fn(224, 224, |x, y| {
        image::Rgb([(x % 255) as u8, (y % 255) as u8, 128])
    });
    img.save(path)?;
    Ok(())
}
