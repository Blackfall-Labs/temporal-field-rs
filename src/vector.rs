//! Field Vector - Signed intensity vector for temporal fields
//!
//! The atomic unit of storage in temporal fields. Each value is stored as
//! a signed byte (-100 to +100) representing intensity from -1.0 to +1.0.

use std::ops::Range;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Signed intensity vector for temporal fields.
///
/// Each value stored as i8 (-100 to +100), representing -1.00 to +1.00.
/// This encodes both polarity (direction) and magnitude (intensity).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FieldVector {
    values: Vec<i8>,
}

impl FieldVector {
    /// Create a new zero-initialized vector.
    pub fn new(dims: usize) -> Self {
        Self {
            values: vec![0; dims],
        }
    }

    /// Create from raw i8 values.
    pub fn from_raw(values: Vec<i8>) -> Self {
        Self { values }
    }

    /// Create from f32 slice (clamped and quantized).
    pub fn from_f32_slice(values: &[f32]) -> Self {
        let mut result = Self::new(values.len());
        for (i, &v) in values.iter().enumerate() {
            result.set(i, v);
        }
        result
    }

    /// Get dimensions.
    #[inline]
    pub fn dims(&self) -> usize {
        self.values.len()
    }

    /// Get value at index as f32.
    #[inline]
    pub fn get(&self, idx: usize) -> f32 {
        self.values[idx] as f32 / 100.0
    }

    /// Set value at index (clamped to [-1.0, +1.0]).
    #[inline]
    pub fn set(&mut self, idx: usize, value: f32) {
        let clamped = value.clamp(-1.0, 1.0);
        self.values[idx] = (clamped * 100.0).round() as i8;
    }

    /// Get raw i8 value.
    #[inline]
    pub fn get_raw(&self, idx: usize) -> i8 {
        self.values[idx]
    }

    /// Decay all values toward zero.
    pub fn decay(&mut self, retention: f32) {
        for v in &mut self.values {
            let current = *v as f32 / 100.0;
            let decayed = current * retention;
            *v = (decayed * 100.0).round() as i8;
        }
    }

    /// Add another vector (saturating).
    pub fn add(&mut self, other: &FieldVector) {
        debug_assert_eq!(self.dims(), other.dims());
        for i in 0..self.values.len() {
            let sum = (self.values[i] as i16) + (other.values[i] as i16);
            self.values[i] = sum.clamp(-100, 100) as i8;
        }
    }

    /// Add f32 values to a range (saturating).
    pub fn add_to_range(&mut self, values: &[f32], range: Range<usize>) {
        let range_len = range.len();
        for (i, &v) in values.iter().take(range_len).enumerate() {
            let idx = range.start + i;
            if idx < self.values.len() {
                let current = self.values[idx] as i16;
                let delta = (v.clamp(-1.0, 1.0) * 100.0).round() as i16;
                self.values[idx] = (current + delta).clamp(-100, 100) as i8;
            }
        }
    }

    /// Set values in a range.
    pub fn set_range(&mut self, values: &[f32], range: Range<usize>) {
        let range_len = range.len();
        for (i, &v) in values.iter().take(range_len).enumerate() {
            let idx = range.start + i;
            if idx < self.values.len() {
                self.set(idx, v);
            }
        }
    }

    /// Get values from a range.
    pub fn get_range(&self, range: Range<usize>) -> Vec<f32> {
        (range.start..range.end.min(self.dims()))
            .map(|i| self.get(i))
            .collect()
    }

    /// Compute energy (sum of squares) in a range.
    pub fn range_energy(&self, range: Range<usize>) -> f32 {
        (range.start..range.end.min(self.dims()))
            .map(|i| {
                let v = self.get(i);
                v * v
            })
            .sum()
    }

    /// Check if range is active (energy above threshold).
    pub fn range_active(&self, range: Range<usize>, threshold: f32) -> bool {
        self.range_energy(range) > threshold
    }

    /// Check if all values are zero.
    pub fn is_zero(&self) -> bool {
        self.values.iter().all(|&v| v == 0)
    }

    /// Count non-zero values.
    pub fn non_zero_count(&self) -> usize {
        self.values.iter().filter(|&&v| v != 0).count()
    }

    /// Get maximum absolute value.
    pub fn max_abs(&self) -> f32 {
        self.values.iter().map(|&v| v.abs()).max().unwrap_or(0) as f32 / 100.0
    }

    /// Scale all values by factor.
    pub fn scale(&mut self, factor: f32) {
        for v in &mut self.values {
            let scaled = (*v as f32 / 100.0) * factor;
            *v = (scaled.clamp(-1.0, 1.0) * 100.0).round() as i8;
        }
    }

    /// Compute L2 norm.
    pub fn norm(&self) -> f32 {
        self.values
            .iter()
            .map(|&v| {
                let f = v as f32 / 100.0;
                f * f
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Dot product with another vector.
    pub fn dot(&self, other: &FieldVector) -> f32 {
        debug_assert_eq!(self.dims(), other.dims());
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(&a, &b)| (a as f32 / 100.0) * (b as f32 / 100.0))
            .sum()
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
    fn test_set_get() {
        let mut v = FieldVector::new(64);
        v.set(0, 0.75);
        v.set(10, -0.5);

        assert!((v.get(0) - 0.75).abs() < 0.01);
        assert!((v.get(10) - -0.5).abs() < 0.01);
    }

    #[test]
    fn test_clamp() {
        let mut v = FieldVector::new(64);
        v.set(0, 1.5); // Should clamp to 1.0
        v.set(1, -2.0); // Should clamp to -1.0

        assert!((v.get(0) - 1.0).abs() < 0.01);
        assert!((v.get(1) - -1.0).abs() < 0.01);
    }

    #[test]
    fn test_decay() {
        let mut v = FieldVector::new(64);
        v.set(0, 1.0);
        v.set(1, -1.0);

        v.decay(0.5);

        assert!((v.get(0) - 0.5).abs() < 0.01);
        assert!((v.get(1) - -0.5).abs() < 0.01);
    }

    #[test]
    fn test_range_energy() {
        let mut v = FieldVector::new(128);
        v.set_range(&vec![0.5; 32], 0..32);

        let energy = v.range_energy(0..32);
        // 32 values of 0.5^2 = 32 * 0.25 = 8.0
        assert!((energy - 8.0).abs() < 0.5);

        // Other range should have zero energy
        assert!(v.range_energy(64..96).abs() < 0.01);
    }
}
