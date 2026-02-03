//! Field configuration
//!
//! ASTRO_004 compliant: No floats. Uses u8 for retention (255 = 1.0).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for a temporal field.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FieldConfig {
    /// Number of dimensions per frame.
    pub dims: usize,

    /// Number of frames in the ring buffer.
    pub frame_count: usize,

    /// Decay retention per tick (0 = instant zero, 255 = no decay).
    /// Scale: 255 = 1.0, 230 ≈ 0.90, 128 = 0.50
    pub retention: u8,

    /// Tick rate in Hz (for time calculations).
    pub tick_rate_hz: u32,
}

impl FieldConfig {
    /// Create a standard configuration.
    /// retention: u8 where 255 = 1.0 (no decay), 230 ≈ 0.90
    pub fn new(dims: usize, frame_count: usize, retention: u8) -> Self {
        Self {
            dims,
            frame_count,
            retention,
            tick_rate_hz: 100,
        }
    }

    /// Get temporal window duration in milliseconds.
    pub fn window_ms(&self) -> u32 {
        (self.frame_count as u32 * 1000) / self.tick_rate_hz
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.dims == 0 {
            return Err("dims must be > 0");
        }
        if self.frame_count == 0 {
            return Err("frame_count must be > 0");
        }
        // retention is u8, always valid (0-255)
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = FieldConfig::new(64, 10, 242); // 242 ≈ 0.95
        assert_eq!(config.dims, 64);
        assert_eq!(config.frame_count, 10);
        assert_eq!(config.retention, 242);
    }

    #[test]
    fn test_window_ms() {
        let config = FieldConfig::new(64, 50, 255);
        // 50 frames at 100Hz = 500ms
        assert_eq!(config.window_ms(), 500);
    }

    #[test]
    fn test_validate() {
        let valid = FieldConfig::new(64, 10, 242);
        assert!(valid.validate().is_ok());

        let invalid_dims = FieldConfig::new(0, 10, 242);
        assert!(invalid_dims.validate().is_err());

        let invalid_frames = FieldConfig::new(64, 0, 242);
        assert!(invalid_frames.validate().is_err());
    }
}
