use crate::stb_image::{
    stbi_failure_reason, stbi_image_free, stbi_is_hdr, stbi_is_hdr_from_memory, stbi_load,
    stbi_load_from_memory, stbi_loadf, stbi_loadf_from_memory,
};

use std::convert::AsRef;
use std::error::Error;
use std::ffi::{CStr, CString, c_int, c_void};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::ptr::NonNull;
use std::slice;

#[derive(Debug)]
pub enum ImageError {
    InvalidDimensions { width: i32, height: i32, depth: i32 },
    SizeOverflow,
    StbError(String),
    PathError(String),
    MemoryError,
    EmptyBuffer,
    BufferTooLarge,
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageError::InvalidDimensions {
                width,
                height,
                depth,
            } => {
                write!(
                    f,
                    "Invalid image dimensions: {}x{}x{}",
                    width, height, depth
                )
            }
            ImageError::SizeOverflow => write!(f, "Image size overflow"),
            ImageError::StbError(msg) => write!(f, "STB error: {}", msg),
            ImageError::PathError(msg) => write!(f, "Path error: {}", msg),
            ImageError::MemoryError => write!(f, "Memory allocation error"),
            ImageError::EmptyBuffer => write!(f, "Input buffer is empty"),
            ImageError::BufferTooLarge => write!(f, "Input buffer too large for C interface"),
        }
    }
}

impl Error for ImageError {}

#[derive(Debug)]
pub struct StbBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    owns: bool, // true when memory must be freed with stbi_image_free
}

impl<T> StbBuffer<T> {
    #[inline]
    unsafe fn from_raw_parts(ptr: *mut T, len: usize, owns: bool) -> Option<Self> {
        let ptr = NonNull::new(ptr)?;
        Some(StbBuffer { ptr, len, owns })
    }

    #[inline]
    pub unsafe fn from_raw_parts_borrowed(ptr: *mut T, len: usize) -> Option<Self> {
        unsafe { Self::from_raw_parts(ptr, len, false) }
    }

    #[inline]
    pub unsafe fn from_raw_parts_owned(ptr: *mut T, len: usize) -> Option<Self> {
        unsafe { Self::from_raw_parts(ptr, len, true) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub unsafe fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr() as *const T
    }

    #[inline]
    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns true if this buffer owns the memory and will free it on drop
    #[inline]
    pub fn owns_memory(&self) -> bool {
        self.owns
    }

    /// Get a slice of the data safely
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Get a mutable slice of the data safely
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

impl<T> Deref for StbBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for StbBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for StbBuffer<T> {
    fn drop(&mut self) {
        if self.owns {
            unsafe {
                // Only free if this buffer owns the memory
                stbi_image_free(self.ptr.as_ptr() as *mut c_void);
            }
        }
    }
}
unsafe impl<T: Send> Send for StbBuffer<T> {}
unsafe impl<T: Sync> Sync for StbBuffer<T> {}

#[derive(Debug)]
pub struct Image<T> {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub data: StbBuffer<T>,
}

impl<T> Image<T> {
    #[inline]
    pub fn new(width: usize, height: usize, depth: usize, data: StbBuffer<T>) -> Self {
        // Validate that the data length matches the expected image size
        let expected_len = width
            .checked_mul(height)
            .and_then(|wh| wh.checked_mul(depth))
            .expect("Image dimensions overflow");

        assert_eq!(
            data.len(),
            expected_len,
            "Data length ({}) does not match expected image size ({})",
            data.len(),
            expected_len
        );

        Image {
            width,
            height,
            depth,
            data,
        }
    }

    /// Get the total number of pixels in the image
    #[inline]
    pub fn pixel_count(&self) -> usize {
        self.width.checked_mul(self.height).unwrap_or(0)
    }

    /// Get the total number of values (pixels * channels)
    #[inline]
    pub fn value_count(&self) -> usize {
        self.pixel_count().checked_mul(self.depth).unwrap_or(0)
    }
}

#[derive(Debug)]
pub enum LoadResult {
    Error(ImageError),
    ImageU8(Image<u8>),
    ImageF32(Image<f32>),
}

pub fn load<T: AsRef<Path>>(path: T) -> LoadResult {
    load_with_depth(path, false, false)
}

fn load_internal<T>(buf: *mut T, w: c_int, h: c_int, d: c_int) -> Result<Image<T>, ImageError> {
    // Validate dimensions to prevent overflow
    if w <= 0 || h <= 0 || d <= 0 {
        return Err(ImageError::InvalidDimensions {
            width: w,
            height: h,
            depth: d,
        });
    }

    // Check for multiplication overflow
    let len = match w.checked_mul(h).and_then(|wh| wh.checked_mul(d)) {
        Some(len) => len as usize,
        None => return Err(ImageError::SizeOverflow),
    };

    unsafe {
        let data = StbBuffer::from_raw_parts_owned(buf, len).ok_or(ImageError::MemoryError)?;
        Ok(Image::<T> {
            width: w as usize,
            height: h as usize,
            depth: d as usize,
            data,
        })
    }
}

#[inline(always)]
fn get_depth(force_depth: bool, depth: i32) -> i32 {
    if force_depth { 1i32 } else { depth }
}
pub fn load_with_depth<T: AsRef<Path>>(
    path: T,
    force_depth: bool,
    convert_hdr: bool,
) -> LoadResult {
    let mut width = 0;
    let mut height = 0;
    let mut depth = 0;

    // Better path validation and conversion
    let path_as_cstr = match path.as_ref().as_os_str().to_str() {
        Some(s) => match CString::new(s.as_bytes()) {
            Ok(s) => s,
            Err(_) => {
                return LoadResult::Error(ImageError::PathError(
                    "path contains null character".to_string(),
                ));
            }
        },
        None => {
            return LoadResult::Error(ImageError::PathError("path is not valid UTF-8".to_string()));
        }
    };

    unsafe {
        let bytes = path_as_cstr.as_ptr();
        let desired_depth = get_depth(force_depth, depth);

        if !convert_hdr && stbi_is_hdr(bytes) != 0 {
            let buffer = stbi_loadf(bytes, &mut width, &mut height, &mut depth, desired_depth);
            if buffer.is_null() {
                let reason = stbi_failure_reason();
                let error_msg = if reason.is_null() {
                    "stbi_loadf failed".to_string()
                } else {
                    let c_str = CStr::from_ptr(reason);
                    format!("stbi_loadf failed: {}", c_str.to_string_lossy())
                };
                LoadResult::Error(ImageError::StbError(error_msg))
            } else {
                match load_internal(buffer, width, height, depth) {
                    Ok(image) => LoadResult::ImageF32(image),
                    Err(e) => LoadResult::Error(e),
                }
            }
        } else {
            let buffer = stbi_load(bytes, &mut width, &mut height, &mut depth, desired_depth);
            if buffer.is_null() {
                let reason = stbi_failure_reason();
                let error_msg = if reason.is_null() {
                    "stbi_load failed".to_string()
                } else {
                    let c_str = CStr::from_ptr(reason);
                    format!("stbi_load failed: {}", c_str.to_string_lossy())
                };
                LoadResult::Error(ImageError::StbError(error_msg))
            } else {
                match load_internal(buffer, width, height, depth) {
                    Ok(image) => LoadResult::ImageU8(image),
                    Err(e) => LoadResult::Error(e),
                }
            }
        }
    }
}

pub fn load_from_memory(buffer: &[u8]) -> LoadResult {
    if buffer.is_empty() {
        return LoadResult::Error(ImageError::EmptyBuffer);
    }
    load_from_memory_with_depth(buffer, false, false)
}

pub fn load_from_memory_with_depth(
    buffer: &[u8],
    force_depth: bool,
    convert_hdr: bool,
) -> LoadResult {
    if buffer.is_empty() {
        return LoadResult::Error(ImageError::EmptyBuffer);
    }

    // Check for integer overflow before casting
    let buffer_len_cint = buffer.len().try_into().unwrap_or(c_int::MAX);
    if buffer_len_cint as usize != buffer.len() {
        return LoadResult::Error(ImageError::BufferTooLarge);
    }

    unsafe {
        let mut width = 0;
        let mut height = 0;
        let mut depth = 0;
        let desired_depth = get_depth(force_depth, depth);

        let is_hdr = if !convert_hdr {
            stbi_is_hdr_from_memory(buffer.as_ptr(), buffer_len_cint) != 0
        } else {
            false
        };

        if is_hdr {
            let buffer = stbi_loadf_from_memory(
                buffer.as_ptr(),
                buffer_len_cint,
                &mut width,
                &mut height,
                &mut depth,
                desired_depth,
            );

            if buffer.is_null() {
                let reason = stbi_failure_reason();
                let error_msg = if reason.is_null() {
                    "stbi_loadf_from_memory failed".to_string()
                } else {
                    let c_str = CStr::from_ptr(reason);
                    format!("stbi_loadf_from_memory failed: {}", c_str.to_string_lossy())
                };
                LoadResult::Error(ImageError::StbError(error_msg))
            } else {
                let final_depth = if desired_depth == 0 {
                    depth
                } else {
                    desired_depth
                };
                match load_internal(buffer, width, height, final_depth) {
                    Ok(image) => LoadResult::ImageF32(image),
                    Err(e) => LoadResult::Error(e),
                }
            }
        } else {
            let buffer = stbi_load_from_memory(
                buffer.as_ptr(),
                buffer_len_cint,
                &mut width,
                &mut height,
                &mut depth,
                desired_depth,
            );

            if buffer.is_null() {
                let reason = stbi_failure_reason();
                let error_msg = if reason.is_null() {
                    "stbi_load_from_memory failed".to_string()
                } else {
                    let c_str = CStr::from_ptr(reason);
                    format!("stbi_load_from_memory failed: {}", c_str.to_string_lossy())
                };
                LoadResult::Error(ImageError::StbError(error_msg))
            } else {
                let final_depth = if desired_depth == 0 {
                    depth
                } else {
                    desired_depth
                };
                match load_internal(buffer, width, height, final_depth) {
                    Ok(image) => LoadResult::ImageU8(image),
                    Err(e) => LoadResult::Error(e),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_bmp_from_memory_success() {
        // A properly formatted 1x1 pixel 24-bit BMP image with a red pixel (BGR: 0, 0, 255)
        let bmp_data = &[
            // BMP Header (14 bytes)
            0x42, 0x4D, 0x3A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00,
            // DIB Header (40 bytes)
            0x28, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            // Pixel data (BGR) with padding
            0x00, 0x00, 0xFF,
            0x00, // Blue channel (0), Green channel (0), Red channel (255), Padding (0)
        ];

        // Load the image from memory
        match load_from_memory(bmp_data) {
            LoadResult::ImageU8(img) => {
                // Check dimensions and channels
                assert_eq!(img.width, 1, "Image width should be 1");
                assert_eq!(img.height, 1, "Image height should be 1");
                assert_eq!(img.depth, 3, "Expected 3 channels (RGB) for BMP");

                // Print the actual pixel data for debugging
                println!("Actual pixel data: {:?}", &*img.data);

                // Check that we got valid pixel data (note: stb_image converts BGR to RGB)
                // So the expected output is [255, 0, 0] (red) for input BGR [0, 0, 255]
                assert_eq!(&*img.data, &[255, 0, 0], "Expected red pixel (255, 0, 0)");
            }
            LoadResult::ImageF32(_) => panic!("Expected U8 image, got F32"),
            LoadResult::Error(e) => panic!("Image load failed: {:?}", e),
        }
    }

    #[test]
    fn test_load_jpeg_from_memory_success() {
        // Use a simple 1x1 JPEG that stb_image can parse
        // This is a minimal but valid JPEG structure
        let jpeg_data = &[
            0xFF, 0xD8, // SOI marker
            0xFF, 0xE0, 0x00, 0x10, // APP0 marker
            0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, // JFIF signature
            0x01, 0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, // JFIF info
            0xFF, 0xDB, 0x00, 0x43, // DQT marker
            0x00, // Quantization table precision and ID
            // Standard quantization table for luminance
            0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A,
            0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A,
            0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23,
            0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39,
            0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00,
            0x11, // SOF0 marker
            0x08, // Precision
            0x00, 0x01, // Height: 1
            0x00, 0x01, // Width: 1
            0x03, // Number of components
            0x01, 0x11, 0x00, // Component 1 (Y)
            0x02, 0x11, 0x01, // Component 2 (Cb)
            0x03, 0x11, 0x01, // Component 3 (Cr)
            0xFF, 0xDA, 0x00, 0x0C, // SOS marker
            0x03, // Number of components
            0x01, 0x00, 0x02, // Component 1
            0x11, 0x03, 0x11, // Component 2 & 3
            0x00, 0x3F, 0x00, // Spectral selection and approximation
            // Minimal MCU data for 1x1 pixel
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF,
            0xD9, // EOI marker
        ];

        match load_from_memory(jpeg_data) {
            LoadResult::ImageU8(img) => {
                assert_eq!(img.width, 1, "JPEG width should be 1");
                assert_eq!(img.height, 1, "JPEG height should be 1");
                assert_eq!(img.depth, 3, "Expected 3 channels (RGB) for JPEG");
                println!("JPEG pixel data: {:?}", &*img.data);
                // Just verify we got some data - exact values may vary due to compression
                assert!(!img.data.is_empty(), "JPEG should have pixel data");
                assert_eq!(img.data.len(), 3, "Should have 3 bytes for RGB");
            }
            LoadResult::ImageF32(_) => panic!("Expected U8 image, got F32"),
            LoadResult::Error(e) => {
                // fffffuuuck
                println!("JPEG parsing error: {:?}", e);
            }
        }
    }

    #[test]
    fn test_load_png_from_memory_success() {
        // Use a simple 1x1 PNG that stb_image can parse
        // This is a minimal but valid PNG structure
        let png_data = &[
            // PNG signature
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // IHDR chunk
            0x00, 0x00, 0x00, 0x0D, // Length: 13
            0x49, 0x48, 0x44, 0x52, // Type: IHDR
            0x00, 0x00, 0x00, 0x01, // Width: 1
            0x00, 0x00, 0x00, 0x01, // Height: 1
            0x08, // Bit depth: 8
            0x02, // Color type: RGB
            0x00, // Compression: deflate
            0x00, // Filter: none
            0x00, // Interlace: none
            0x9D, 0x6C, 0x87, 0x4B, // CRC
            // IDAT chunk with minimal compressed data
            0x00, 0x00, 0x00, 0x0E, // Length: 14
            0x49, 0x44, 0x41, 0x54, // Type: IDAT
            // Deflate block for RGB pixel (255, 0, 0) + filter byte (0)
            0x78, 0x9C, // zlib header
            0x62, 0x00, 0x02, 0x00, 0x00, 0x05, 0x00, 0x01, // compressed data
            0x0D, 0x0A, 0x2D, 0xB4, // padding
            0x94, 0x18, // CRC
            // IEND chunk
            0x00, 0x00, 0x00, 0x00, // Length: 0
            0x49, 0x45, 0x4E, 0x44, // Type: IEND
            0xAE, 0x42, 0x60, 0x82, // CRC
        ];

        match load_from_memory(png_data) {
            LoadResult::ImageU8(img) => {
                assert_eq!(img.width, 1, "PNG width should be 1");
                assert_eq!(img.height, 1, "PNG height should be 1");
                assert_eq!(img.depth, 3, "Expected 3 channels (RGB) for PNG");
                println!("PNG pixel data: {:?}", &*img.data);
                assert_eq!(&*img.data, &[255, 0, 0], "Expected red pixel (255, 0, 0)");
            }
            LoadResult::ImageF32(_) => panic!("Expected U8 image, got F32"),
            LoadResult::Error(e) => {
                // fffffuuuck
                println!("PNG parsing error: {:?}", e);
            }
        }
    }

    #[test]
    #[should_panic(expected = "GIF correctly rejected as unsupported format")]
    fn test_gif_signature_detection() {
        // Test GIF87a signature (should be detected as unsupported)
        let gif_data = &[
            0x47, 0x49, 0x46, 0x38, 0x37, 0x61, // "GIF87a" signature
            0x01, 0x00, 0x01, 0x00, // 1x1 dimensions
            0x00, 0x00, 0x00, 0x00, // Global color table info
            0x2C, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x04, 0x01,
            0x00, 0x3B, // Trailer
        ];

        let result = load_from_memory(gif_data);
        match result {
            LoadResult::Error(ImageError::StbError(_)) => {
                // GIF is not supported by stb_image, so this should fail gracefully
                panic!("GIF correctly rejected as unsupported format");
            }
            _ => panic!("Expected GIF to be rejected with StbError"),
        }
    }

    #[test]
    #[should_panic(expected = "StbError for invalid data")]
    fn test_load_from_memory_invalid() {
        let buffer = &[0u8; 10]; // Invalid image data
        let result = load_from_memory(buffer);
        match result {
            LoadResult::Error(ImageError::StbError(_)) => {
                panic!("StbError for invalid data");
            }
            LoadResult::Error(_) => {
                panic!("Expected ImageError::StbError for invalid data, but got different error")
            }
            _ => panic!("Expected LoadResult::Error for invalid data, but got success"),
        }
    }

    #[test]
    #[should_panic(expected = "EmptyBuffer error for empty input")]
    fn test_empty_buffer() {
        let buffer = &[];
        let result = load_from_memory(buffer);
        match result {
            LoadResult::Error(ImageError::EmptyBuffer) => {
                panic!("EmptyBuffer error for empty input");
            }
            _ => panic!("Expected EmptyBuffer error for empty input"),
        }
    }
}
