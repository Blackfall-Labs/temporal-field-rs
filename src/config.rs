//! Substrate configuration

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for a temporal field substrate.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SubstrateConfig {
    /// Number of dimensions per frame.
    pub dims: usize,

    /// Number of frames in the ring buffer.
    pub frame_count: usize,

    /// Decay retention per tick (0.0 = instant zero, 1.0 = no decay).
    pub retention: f32,

    /// Tick rate in Hz (for time calculations).
    pub tick_rate_hz: u32,
}

impl SubstrateConfig {
    /// Create a standard configuration.
    pub fn new(dims: usize, frame_count: usize, retention: f32) -> Self {
        Self {
            dims,
            frame_count,
            retention,
            tick_rate_hz: 100,
        }
    }

    /// Get temporal window duration in milliseconds.
    pub fn window_ms(&self) -> f32 {
        (self.frame_count as f32 * 1000.0) / self.tick_rate_hz as f32
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.dims == 0 {
            return Err("dims must be > 0");
        }
        if self.frame_count == 0 {
            return Err("frame_count must be > 0");
        }
        if !(0.0..=1.0).contains(&self.retention) {
            return Err("retention must be in [0, 1]");
        }
        Ok(())
    }
}
