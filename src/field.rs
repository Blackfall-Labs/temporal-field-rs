//! Temporal Field - the core ring buffer with decay

use crate::config::FieldConfig;
use crate::vector::FieldVector;
use std::ops::Range;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The core temporal field.
///
/// A ring buffer of vectors where all values decay each tick.
/// This is the shared foundation for:
/// - BindingField (sensory pattern integration)
/// - ConvergenceField (mesh output integration)
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TemporalField {
    /// Ring buffer of frames.
    frames: Vec<FieldVector>,

    /// Configuration.
    config: FieldConfig,

    /// Current write position.
    write_head: usize,

    /// Total ticks elapsed.
    tick_count: u64,
}

impl TemporalField {
    /// Create a new field with the given configuration.
    pub fn new(config: FieldConfig) -> Self {
        let frames = (0..config.frame_count)
            .map(|_| FieldVector::new(config.dims))
            .collect();

        Self {
            frames,
            config,
            write_head: 0,
            tick_count: 0,
        }
    }

    // =========================================================================
    // TIME ADVANCEMENT
    // =========================================================================

    /// Advance time by one tick - decay all frames.
    pub fn tick(&mut self) {
        self.tick_count += 1;
        for frame in &mut self.frames {
            frame.decay(self.config.retention);
        }
    }

    /// Advance multiple ticks.
    pub fn tick_n(&mut self, n: usize) {
        for _ in 0..n {
            self.tick();
        }
    }

    /// Advance write head to next frame.
    pub fn advance_write_head(&mut self) {
        self.write_head = (self.write_head + 1) % self.config.frame_count;
    }

    // =========================================================================
    // WRITING
    // =========================================================================

    /// Write values to a region of the current frame (additive).
    pub fn write_region(&mut self, values: &[f32], range: Range<usize>) {
        self.frames[self.write_head].add_to_range(values, range);
    }

    /// Set values in a region of the current frame (replace, not add).
    pub fn set_region(&mut self, values: &[f32], range: Range<usize>) {
        self.frames[self.write_head].set_range(values, range);
    }

    /// Add a full vector to current frame.
    pub fn write_full(&mut self, vector: &FieldVector) {
        self.frames[self.write_head].add(vector);
    }

    /// Clear the current frame.
    pub fn clear_current(&mut self) {
        self.frames[self.write_head] = FieldVector::new(self.config.dims);
    }

    // =========================================================================
    // READING
    // =========================================================================

    /// Read the current frame.
    pub fn read_current(&self) -> &FieldVector {
        &self.frames[self.write_head]
    }

    /// Read a specific region from current frame.
    pub fn read_region(&self, range: Range<usize>) -> Vec<f32> {
        self.frames[self.write_head].get_range(range)
    }

    /// Get energy in a region of current frame.
    pub fn region_energy(&self, range: Range<usize>) -> f32 {
        self.frames[self.write_head].range_energy(range)
    }

    /// Check if region is active (energy above threshold).
    pub fn region_active(&self, range: Range<usize>, threshold: f32) -> bool {
        self.region_energy(range) > threshold
    }

    /// Read the last N frames in chronological order (oldest first).
    pub fn read_window(&self, n: usize) -> Vec<&FieldVector> {
        let n = n.min(self.config.frame_count);
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let idx = (self.write_head + self.config.frame_count - n + 1 + i)
                % self.config.frame_count;
            result.push(&self.frames[idx]);
        }

        result
    }

    /// Get peak values in a region over the last N frames.
    pub fn region_peak(&self, range: Range<usize>, window: usize) -> Vec<f32> {
        let frames = self.read_window(window);
        if frames.is_empty() {
            return vec![0.0; range.len()];
        }

        let mut best_frame_idx = 0;
        let mut best_energy = 0.0f32;

        for (i, frame) in frames.iter().enumerate() {
            let energy = frame.range_energy(range.clone());
            if energy > best_energy {
                best_energy = energy;
                best_frame_idx = i;
            }
        }

        frames[best_frame_idx].get_range(range)
    }

    /// Get mean values in a region over the last N frames.
    pub fn region_mean(&self, range: Range<usize>, window: usize) -> Vec<f32> {
        let frames = self.read_window(window);
        if frames.is_empty() {
            return vec![0.0; range.len()];
        }

        let len = range.len();
        let mut avg = vec![0.0; len];

        for frame in &frames {
            for (i, idx) in range.clone().enumerate() {
                avg[i] += frame.get(idx);
            }
        }

        let n = frames.len() as f32;
        for v in &mut avg {
            *v /= n;
        }

        avg
    }

    // =========================================================================
    // METRICS
    // =========================================================================

    /// Get configuration.
    pub fn config(&self) -> &FieldConfig {
        &self.config
    }

    /// Get current tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Get write head position.
    pub fn write_head(&self) -> usize {
        self.write_head
    }

    /// Get total dimensions.
    pub fn dims(&self) -> usize {
        self.config.dims
    }

    /// Get frame count.
    pub fn frame_count(&self) -> usize {
        self.config.frame_count
    }

    /// Get maximum absolute value in field.
    pub fn max_energy(&self) -> f32 {
        self.frames
            .iter()
            .map(|f| f.max_abs())
            .fold(0.0f32, f32::max)
    }

    /// Get total non-zero count.
    pub fn total_activity(&self) -> usize {
        self.frames.iter().map(|f| f.non_zero_count()).sum()
    }

    /// Clear entire field.
    pub fn clear(&mut self) {
        for frame in &mut self.frames {
            *frame = FieldVector::new(self.config.dims);
        }
        self.write_head = 0;
        self.tick_count = 0;
    }

    /// Convert tick difference to milliseconds.
    pub fn ticks_to_ms(&self, ticks: u64) -> f32 {
        (ticks as f32 * 1000.0) / self.config.tick_rate_hz as f32
    }

    /// Convert milliseconds to ticks.
    pub fn ms_to_ticks(&self, ms: f32) -> u64 {
        ((ms * self.config.tick_rate_hz as f32) / 1000.0).round() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_substrate() {
        let config = FieldConfig::new(64, 10, 0.95);
        let substrate = TemporalField::new(config);

        assert_eq!(substrate.dims(), 64);
        assert_eq!(substrate.frame_count(), 10);
        assert_eq!(substrate.tick_count(), 0);
        assert_eq!(substrate.total_activity(), 0);
    }

    #[test]
    fn test_write_and_read_region() {
        let config = FieldConfig::new(128, 10, 0.95);
        let mut substrate = TemporalField::new(config);

        let values = vec![0.5; 32];
        substrate.write_region(&values, 0..32);

        assert!(substrate.region_active(0..32, 0.1));
        assert!(!substrate.region_active(32..64, 0.1));
    }

    #[test]
    fn test_decay() {
        let config = FieldConfig::new(64, 10, 0.5);
        let mut substrate = TemporalField::new(config);

        substrate.write_region(&vec![1.0; 64], 0..64);
        let initial = substrate.region_energy(0..64);

        substrate.tick();
        let after_tick = substrate.region_energy(0..64);

        // Energy should be ~0.25 of initial (0.5^2 per value, summed)
        assert!(after_tick < initial * 0.5);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let config = FieldConfig::new(64, 3, 1.0); // No decay
        let mut substrate = TemporalField::new(config);

        // Write 5 frames (should wrap around)
        for i in 0..5 {
            substrate.clear_current();
            substrate.write_region(&vec![(i + 1) as f32 * 0.1; 64], 0..64);
            substrate.advance_write_head();
        }

        // Current position should be 5 % 3 = 2
        assert_eq!(substrate.write_head(), 2);
    }

    #[test]
    fn test_window_chronological() {
        let config = FieldConfig::new(64, 5, 1.0);
        let mut substrate = TemporalField::new(config);

        // Write 3 distinct values
        for i in 0..3 {
            substrate.clear_current();
            substrate.write_region(&vec![(i + 1) as f32 * 0.25; 1], 0..1);
            substrate.advance_write_head();
        }

        let window = substrate.read_window(3);
        assert_eq!(window.len(), 3);

        // Should be chronological (oldest first)
        assert!((window[0].get(0) - 0.25).abs() < 0.01);
        assert!((window[1].get(0) - 0.50).abs() < 0.01);
        assert!((window[2].get(0) - 0.75).abs() < 0.01);
    }
}
