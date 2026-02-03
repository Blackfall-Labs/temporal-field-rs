//! Field Vector - Signal-based storage for temporal fields
//!
//! ASTRO_004 compliant: Uses Signal (polarity + magnitude) throughout.
//! No floats in neural computation paths.

use std::ops::Range;
use ternsig::Signal;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Signal-based vector for temporal fields.
///
/// Each element is a Signal (polarity: i8, magnitude: u8).
/// Total size: 2 bytes per element.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FieldVector {
    signals: Vec<Signal>,
}

impl FieldVector {
    /// Create a new zero-initialized vector.
    pub fn new(dims: usize) -> Self {
        Self {
            signals: vec![Signal::ZERO; dims],
        }
    }

    /// Create from raw Signal slice.
    pub fn from_signals(signals: Vec<Signal>) -> Self {
        Self { signals }
    }

    /// Get dimensions.
    #[inline]
    pub fn dims(&self) -> usize {
        self.signals.len()
    }

    /// Get Signal at index.
    #[inline]
    pub fn get(&self, idx: usize) -> Signal {
        self.signals[idx]
    }

    /// Set Signal at index.
    #[inline]
    pub fn set(&mut self, idx: usize, signal: Signal) {
        self.signals[idx] = signal;
    }

    /// Get as signed i16 (polarity × magnitude) for arithmetic.
    #[inline]
    pub fn get_i16(&self, idx: usize) -> i16 {
        let s = self.signals[idx];
        (s.polarity as i16) * (s.magnitude as i16)
    }

    /// Set from signed i16 value (clamped to ±255).
    #[inline]
    pub fn set_i16(&mut self, idx: usize, value: i16) {
        self.signals[idx] = Signal::from_signed_i32(value as i32);
    }

    /// Decay all values toward zero.
    /// retention: u8 where 255 = 1.0 (no decay), 230 ≈ 0.90
    pub fn decay(&mut self, retention: u8) {
        for s in &mut self.signals {
            // magnitude = magnitude * retention / 255
            let new_mag = ((s.magnitude as u16) * (retention as u16) / 255) as u8;
            if new_mag == 0 {
                *s = Signal::ZERO;
            } else {
                s.magnitude = new_mag;
            }
        }
    }

    /// Add another vector (saturating).
    pub fn add(&mut self, other: &FieldVector) {
        debug_assert_eq!(self.dims(), other.dims());
        for i in 0..self.signals.len() {
            let a = self.get_i16(i);
            let b = other.get_i16(i);
            self.set_i16(i, a.saturating_add(b));
        }
    }

    /// Add Signals to a range (saturating).
    pub fn add_to_range(&mut self, signals: &[Signal], range: Range<usize>) {
        let range_len = range.len();
        for (i, &s) in signals.iter().take(range_len).enumerate() {
            let idx = range.start + i;
            if idx < self.signals.len() {
                let current = self.get_i16(idx);
                let delta = (s.polarity as i16) * (s.magnitude as i16);
                self.set_i16(idx, current.saturating_add(delta));
            }
        }
    }

    /// Set Signals in a range.
    pub fn set_range(&mut self, signals: &[Signal], range: Range<usize>) {
        let range_len = range.len();
        for (i, &s) in signals.iter().take(range_len).enumerate() {
            let idx = range.start + i;
            if idx < self.signals.len() {
                self.signals[idx] = s;
            }
        }
    }

    /// Get Signals from a range.
    pub fn get_range(&self, range: Range<usize>) -> Vec<Signal> {
        (range.start..range.end.min(self.dims()))
            .map(|i| self.signals[i])
            .collect()
    }

    /// Compute energy (sum of squared magnitudes) in a range.
    /// Returns u32 to prevent overflow.
    pub fn range_energy(&self, range: Range<usize>) -> u32 {
        (range.start..range.end.min(self.dims()))
            .map(|i| {
                let m = self.signals[i].magnitude as u32;
                m * m
            })
            .sum()
    }

    /// Check if range is active (energy above threshold).
    /// threshold is in squared magnitude units (e.g., 6500 ≈ 1.0 for 64 dims)
    pub fn range_active(&self, range: Range<usize>, threshold: u32) -> bool {
        self.range_energy(range) > threshold
    }

    /// Check if all signals are zero.
    pub fn is_zero(&self) -> bool {
        self.signals.iter().all(|s| s.magnitude == 0)
    }

    /// Count non-zero signals.
    pub fn non_zero_count(&self) -> usize {
        self.signals.iter().filter(|s| s.magnitude > 0).count()
    }

    /// Get maximum magnitude.
    pub fn max_magnitude(&self) -> u8 {
        self.signals.iter().map(|s| s.magnitude).max().unwrap_or(0)
    }

    /// Scale all magnitudes by factor (u8 where 255 = 1.0).
    pub fn scale(&mut self, factor: u8) {
        for s in &mut self.signals {
            s.magnitude = ((s.magnitude as u16) * (factor as u16) / 255) as u8;
        }
    }

    /// Get slice reference for direct access.
    pub fn as_slice(&self) -> &[Signal] {
        &self.signals
    }

    /// Get mutable slice reference.
    pub fn as_mut_slice(&mut self) -> &mut [Signal] {
        &mut self.signals
    }
}

impl Default for FieldVector {
    fn default() -> Self {
        Self::new(64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_zero() {
        let v = FieldVector::new(128);
        assert!(v.is_zero());
        assert_eq!(v.non_zero_count(), 0);
    }

    #[test]
    fn test_set_get_signal() {
        let mut v = FieldVector::new(64);
        v.set(0, Signal::positive(200));
        v.set(10, Signal::negative(128));

        assert_eq!(v.get(0).polarity, 1);
        assert_eq!(v.get(0).magnitude, 200);
        assert_eq!(v.get(10).polarity, -1);
        assert_eq!(v.get(10).magnitude, 128);
    }

    #[test]
    fn test_i16_interface() {
        let mut v = FieldVector::new(64);
        v.set_i16(0, 200);  // positive
        v.set_i16(1, -128); // negative

        assert_eq!(v.get_i16(0), 200);
        assert_eq!(v.get_i16(1), -128);
    }

    #[test]
    fn test_decay() {
        let mut v = FieldVector::new(64);
        v.set(0, Signal::positive(255));
        v.set(1, Signal::negative(255));

        v.decay(128); // ~50% retention

        // 255 * 128 / 255 = 128
        assert_eq!(v.get(0).magnitude, 128);
        assert_eq!(v.get(1).magnitude, 128);
        // Polarity preserved
        assert_eq!(v.get(0).polarity, 1);
        assert_eq!(v.get(1).polarity, -1);
    }

    #[test]
    fn test_range_energy() {
        let mut v = FieldVector::new(128);
        // Set 32 signals with magnitude 128
        for i in 0..32 {
            v.set(i, Signal::positive(128));
        }

        let energy = v.range_energy(0..32);
        // 32 * 128^2 = 32 * 16384 = 524288
        assert_eq!(energy, 524288);

        // Other range should have zero energy
        assert_eq!(v.range_energy(64..96), 0);
    }

    #[test]
    fn test_add_saturating() {
        let mut v = FieldVector::new(4);
        v.set_i16(0, 200);

        let mut other = FieldVector::new(4);
        other.set_i16(0, 100);

        v.add(&other);
        // 200 + 100 = 300, but Signal max is 255
        assert_eq!(v.get(0).magnitude, 255);
    }

    #[test]
    fn test_add_to_range() {
        let mut v = FieldVector::new(64);
        let signals = vec![Signal::positive(100); 8];
        v.add_to_range(&signals, 0..8);

        assert_eq!(v.get(0).magnitude, 100);
        assert_eq!(v.get(7).magnitude, 100);
        assert_eq!(v.get(8).magnitude, 0);
    }

    #[test]
    fn test_set_range() {
        let mut v = FieldVector::new(64);
        let signals = vec![Signal::negative(50); 4];
        v.set_range(&signals, 10..14);

        assert_eq!(v.get(9).magnitude, 0);
        assert_eq!(v.get(10).magnitude, 50);
        assert_eq!(v.get(10).polarity, -1);
        assert_eq!(v.get(13).magnitude, 50);
        assert_eq!(v.get(14).magnitude, 0);
    }

    #[test]
    fn test_get_range() {
        let mut v = FieldVector::new(64);
        v.set(5, Signal::positive(100));
        v.set(6, Signal::negative(200));

        let range = v.get_range(5..7);
        assert_eq!(range.len(), 2);
        assert_eq!(range[0].magnitude, 100);
        assert_eq!(range[1].magnitude, 200);
    }

    #[test]
    fn test_scale() {
        let mut v = FieldVector::new(4);
        v.set(0, Signal::positive(200));
        v.set(1, Signal::negative(100));

        v.scale(128); // ~50%

        // 200 * 128 / 255 ≈ 100
        assert_eq!(v.get(0).magnitude, 100);
        // 100 * 128 / 255 ≈ 50
        assert_eq!(v.get(1).magnitude, 50);
    }
}
