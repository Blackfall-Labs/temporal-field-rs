//! Field Vector - Signal-based storage for temporal fields
//!
//! ASTRO_004 compliant: Uses Signal (polarity × magnitude × multiplier) throughout.
//! No floats in neural computation paths.

use std::ops::Range;
use ternary_signal::Signal;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Signal-based vector for temporal fields.
///
/// Each element is a Signal (polarity: i8, magnitude: u8, multiplier: u8).
/// Total size: 3 bytes per element.
///
/// Arithmetic operations use the full effective value: `polarity × magnitude × multiplier`
/// (range ±65,025). Results are decomposed back into (p, m, k) via `Signal::from_current`.
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

    /// Get the full effective value: `polarity × magnitude × multiplier` (±65,025).
    #[inline]
    pub fn get_current(&self, idx: usize) -> i32 {
        self.signals[idx].current()
    }

    /// Set from a signed i32 value using the full p×m×k range (±65,025).
    #[inline]
    pub fn set_current(&mut self, idx: usize, value: i32) {
        self.signals[idx] = Signal::from_current(value);
    }

    /// Get as signed i16 (polarity × magnitude only, ignores multiplier).
    ///
    /// **Deprecated in favor of [`get_current`]** which uses the full range.
    /// Retained for backward compatibility with code that needs the narrow range.
    #[inline]
    pub fn get_i16(&self, idx: usize) -> i16 {
        let s = self.signals[idx];
        (s.polarity as i16) * (s.magnitude as i16)
    }

    /// Set from signed i16 value (clamped to ±255, multiplier=1).
    ///
    /// **Deprecated in favor of [`set_current`]** which uses the full range.
    /// Retained for backward compatibility.
    #[inline]
    pub fn set_i16(&mut self, idx: usize, value: i16) {
        self.signals[idx] = Signal::from_signed_i32(value as i32);
    }

    /// Decay all values toward zero.
    /// retention: u8 where 255 = 1.0 (no decay), 230 ≈ 0.90
    ///
    /// Decays the effective value (p×m×k), then re-encodes into Signal.
    /// This preserves the full dynamic range during decay.
    pub fn decay(&mut self, retention: u8) {
        for s in &mut self.signals {
            let current = s.current();
            if current == 0 {
                continue;
            }
            // Apply retention to the effective value
            let decayed = (current as i64 * retention as i64 / 255) as i32;
            if decayed == 0 {
                *s = Signal::ZERO;
            } else {
                *s = Signal::from_current(decayed);
            }
        }
    }

    /// Add another vector (saturating at ±65,025).
    pub fn add(&mut self, other: &FieldVector) {
        debug_assert_eq!(self.dims(), other.dims());
        for i in 0..self.signals.len() {
            let a = self.get_current(i);
            let b = other.get_current(i);
            let sum = (a as i64 + b as i64).clamp(-65025, 65025) as i32;
            self.set_current(i, sum);
        }
    }

    /// Add Signals to a range (saturating at ±65,025).
    pub fn add_to_range(&mut self, signals: &[Signal], range: Range<usize>) {
        let range_len = range.len();
        for (i, &s) in signals.iter().take(range_len).enumerate() {
            let idx = range.start + i;
            if idx < self.signals.len() {
                let current = self.get_current(idx);
                let delta = s.current();
                let sum = (current as i64 + delta as i64).clamp(-65025, 65025) as i32;
                self.set_current(idx, sum);
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

    /// Compute energy (sum of squared effective magnitudes) in a range.
    /// Returns u64 to prevent overflow (max per element: 65025² ≈ 4.2B).
    pub fn range_energy(&self, range: Range<usize>) -> u64 {
        (range.start..range.end.min(self.dims()))
            .map(|i| {
                let eff = self.signals[i].effective_magnitude() as u64;
                eff * eff
            })
            .sum()
    }

    /// Check if range is active (energy above threshold).
    pub fn range_active(&self, range: Range<usize>, threshold: u64) -> bool {
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

    /// Get maximum effective magnitude.
    pub fn max_magnitude(&self) -> u16 {
        self.signals.iter().map(|s| s.effective_magnitude()).max().unwrap_or(0)
    }

    /// Scale all values by factor (u8 where 255 = 1.0).
    pub fn scale(&mut self, factor: u8) {
        for s in &mut self.signals {
            let current = s.current();
            let scaled = (current as i64 * factor as i64 / 255) as i32;
            *s = Signal::from_current(scaled);
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
    fn test_current_interface() {
        let mut v = FieldVector::new(64);
        v.set_current(0, 5000);  // should decompose into p=1, m×k≈5000
        v.set_current(1, -3000); // should decompose into p=-1, m×k≈3000

        let val0 = v.get_current(0);
        let val1 = v.get_current(1);
        // Allow small rounding error from decomposition
        assert!((val0 - 5000).abs() < 20, "expected ~5000, got {}", val0);
        assert!((val1 + 3000).abs() < 20, "expected ~-3000, got {}", val1);
    }

    #[test]
    fn test_full_range_add() {
        let mut v = FieldVector::new(4);
        // Set a value > 255 — requires multiplier
        v.set(0, Signal::positive_amplified(200, 100)); // 20,000
        let mut other = FieldVector::new(4);
        other.set(0, Signal::positive_amplified(100, 50)); // 5,000

        v.add(&other);
        let result = v.get_current(0);
        // 20000 + 5000 = 25000
        assert!((result - 25000).abs() < 100, "expected ~25000, got {}", result);
    }

    #[test]
    fn test_decay_full_range() {
        let mut v = FieldVector::new(64);
        v.set(0, Signal::positive_amplified(255, 100)); // 25,500
        v.set(1, Signal::negative_amplified(200, 50));  // -10,000

        v.decay(128); // ~50% retention

        let val0 = v.get_current(0);
        let val1 = v.get_current(1);
        // 25500 * 128/255 ≈ 12800
        assert!((val0 - 12800).abs() < 200, "expected ~12800, got {}", val0);
        // -10000 * 128/255 ≈ -5020
        assert!((val1 + 5020).abs() < 200, "expected ~-5020, got {}", val1);
    }

    #[test]
    fn test_range_energy_full() {
        let mut v = FieldVector::new(128);
        // Set signals with multiplier
        for i in 0..4 {
            v.set(i, Signal::positive_amplified(100, 50)); // effective = 5000
        }

        let energy = v.range_energy(0..4);
        // 4 * 5000² = 4 * 25,000,000 = 100,000,000
        assert_eq!(energy, 100_000_000);
    }

    #[test]
    fn test_add_to_range_full() {
        let mut v = FieldVector::new(64);
        let signals = vec![Signal::positive_amplified(100, 10); 4]; // 1000 each
        v.add_to_range(&signals, 0..4);

        let val = v.get_current(0);
        assert!((val - 1000).abs() < 10, "expected ~1000, got {}", val);
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
    fn test_scale_full_range() {
        let mut v = FieldVector::new(4);
        v.set(0, Signal::positive_amplified(200, 100)); // 20000

        v.scale(128); // ~50%

        let val = v.get_current(0);
        // 20000 * 128/255 ≈ 10039
        assert!((val - 10039).abs() < 200, "expected ~10039, got {}", val);
    }

    #[test]
    fn test_saturation_at_max() {
        let mut v = FieldVector::new(4);
        v.set(0, Signal::positive_amplified(255, 255)); // 65025

        let mut other = FieldVector::new(4);
        other.set(0, Signal::positive_amplified(255, 255)); // 65025

        v.add(&other);
        let result = v.get_current(0);
        // 65025 + 65025 = 130050 → clamped to 65025
        assert!(result <= 65025, "should clamp to 65025, got {}", result);
    }

    // Backward compat: i16 interface still works for narrow-range code
    #[test]
    fn test_i16_interface() {
        let mut v = FieldVector::new(64);
        v.set_i16(0, 200);
        v.set_i16(1, -128);

        assert_eq!(v.get_i16(0), 200);
        assert_eq!(v.get_i16(1), -128);
    }
}
