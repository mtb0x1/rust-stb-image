# rust-stb-image

Safe Rust bindings for the excellent [stb_image](https://github.com/nothings/stb) library, providing fast and lightweight image decoding capabilities.

## Overview

This crate offers a Rust interface to stb_image v2.30, supporting popular image formats including JPEG, PNG, BMP, GIF, and more. It's designed for performance and simplicity, making it ideal for applications that need reliable image loading without any dependencies.
Except for compile time (bindgen and cc).

## Features

- **Multiple formats**: JPEG, PNG, BMP, (GIF, TGA, PSD, PIC PNM too maybe, who knows ?)
- **HDR support**: Load high dynamic range images
- **Memory-safe**: Rust wrapper with proper memory management (hum safe'ish more than safe!)
- **No external dependencies**: Pure Rust bindings to C library
- **Flexible loading**: From files or memory buffers

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
stb_image = "2.30.0"
```

## Usage

### Basic Image Loading

```rust
use stb_image::image;

// Load from file
match image::load("path/to/image.png") {
    image::LoadResult::Image(img) => {
        println!("Loaded: {}x{}", img.width, img.height);
    }
    image::LoadResult::Error(e) => {
        eprintln!("Failed to load image: {}", e);
    }
}

// Load from memory
let image_data = std::fs::read("image.jpg")?;
match image::load_from_memory(&image_data) {
    image::LoadResult::Image(img) => {
        // Process image
    }
    image::LoadResult::Error(e) => {
        eprintln!("Failed to load image: {}", e);
    }
}
```

### Advanced Loading Options

```rust
use stb_image::image;

// Force specific color depth and handle HDR
match image::load_with_depth("image.hdr", false, true) {
    image::LoadResult::ImageF32(img) => {
        // HDR image data as f32
    }
    image::LoadResult::ImageU8(img) => {
        // Standard 8-bit image data
    }
    image::LoadResult::Error(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Documentation

No.

## License

DO WHAT EVER YOU WANT WITH IT, I DON'T CARE, YOU GET IT?
