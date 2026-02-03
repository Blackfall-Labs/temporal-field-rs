//! Observer pattern for temporal fields - brain-style pub/sub
//!
//! When a write crosses a threshold, observers are notified automatically.
//! No polling required - sparks propagate.
//!
//! ASTRO_004 compliant: No floats. Energy as u32, thresholds as u32.
//!
//! ## Hysteresis
//!
//! Thresholds use hysteresis to prevent chattering when values hover near
//! the boundary. Two thresholds control state transitions:
//!
//! - **on_threshold** (higher): Energy must exceed this to become active
//! - **off_threshold** (lower): Energy must drop below this to become quiet
//!
//! When energy is between the thresholds, the previous state is maintained.
//! This provides stable edge-triggered semantics without explicit refractory periods.

use std::ops::Range;

/// Event emitted when field activity crosses a threshold
#[derive(Clone, Debug)]
pub enum FieldEvent {
    /// A region became active (energy crossed on_threshold from below)
    RegionActive {
        region: Range<usize>,
        /// Energy as sum of squared magnitudes
        energy: u32,
        /// The on_threshold that was crossed
        threshold: u32,
    },
    /// A region went quiet (energy dropped below off_threshold)
    RegionQuiet {
        region: Range<usize>,
        /// Energy as sum of squared magnitudes
        energy: u32,
        /// The off_threshold that was crossed
        threshold: u32,
    },
    /// Multiple regions active simultaneously (binding opportunity)
    Convergence {
        active_regions: Vec<Range<usize>>,
        /// Total weighted energy (sum of energy × weight for each active region)
        total_energy: u32,
    },
    /// Peak detected in a region (local maximum)
    Peak {
        region: Range<usize>,
        /// Energy as sum of squared magnitudes
        energy: u32,
        tick: u64,
    },
}

/// Observer that receives field events
pub trait FieldObserver: Send + Sync {
    /// Called when a field event occurs
    fn on_event(&self, event: FieldEvent);
}

/// Function-based observer for simple cases
pub struct FnObserver<F: Fn(FieldEvent) + Send + Sync>(pub F);

impl<F: Fn(FieldEvent) + Send + Sync> FieldObserver for FnObserver<F> {
    fn on_event(&self, event: FieldEvent) {
        (self.0)(event);
    }
}

/// Channel-based observer - sends events to a channel
pub struct ChannelObserver {
    sender: std::sync::mpsc::Sender<FieldEvent>,
}

impl ChannelObserver {
    pub fn new(sender: std::sync::mpsc::Sender<FieldEvent>) -> Self {
        Self { sender }
    }
}

impl FieldObserver for ChannelObserver {
    fn on_event(&self, event: FieldEvent) {
        let _ = self.sender.send(event);
    }
}

/// Configuration for what triggers notifications
#[derive(Clone, Debug)]
pub struct TriggerConfig {
    /// Regions to monitor (empty = monitor all dims as one region)
    pub regions: Vec<MonitoredRegion>,
    /// Minimum regions active for convergence event
    pub convergence_threshold: usize,
}

/// Default hysteresis gap as percentage (20 = 20%).
/// off_threshold = on_threshold * (100 - gap) / 100
pub const DEFAULT_HYSTERESIS_GAP: u8 = 20;

/// A region being monitored for activity with hysteresis thresholds.
///
/// ## Hysteresis
///
/// Two thresholds control state transitions to prevent chattering:
/// - `on_threshold`: Energy must exceed this to become active
/// - `off_threshold`: Energy must drop below this to become quiet
///
/// When energy is between the thresholds, the previous state is maintained.
///
/// ## Energy Units
///
/// Energy is sum of squared magnitudes: Σ(magnitude²)
/// For 64 dims with all magnitudes at 128: 64 × 128² = 1,048,576
/// For 64 dims with all magnitudes at 255: 64 × 255² = 4,161,600
#[derive(Clone, Debug)]
pub struct MonitoredRegion {
    /// Name for identification
    pub name: String,
    /// Dimension range
    pub range: Range<usize>,
    /// Energy threshold to enter active state (higher threshold)
    /// Energy = sum of squared magnitudes
    pub on_threshold: u32,
    /// Energy threshold to leave active state (lower threshold)
    pub off_threshold: u32,
    /// Weight for convergence calculation (100 = 1.0×)
    pub weight: u8,
}

impl MonitoredRegion {
    /// Create a new monitored region with automatic hysteresis.
    ///
    /// The off_threshold is set to 80% of the on_threshold by default,
    /// providing a 20% hysteresis gap to prevent chattering.
    ///
    /// threshold: Energy threshold (sum of squared magnitudes)
    pub fn new(name: impl Into<String>, range: Range<usize>, threshold: u32) -> Self {
        Self {
            name: name.into(),
            range,
            on_threshold: threshold,
            off_threshold: threshold * (100 - DEFAULT_HYSTERESIS_GAP as u32) / 100,
            weight: 100,
        }
    }

    /// Create a monitored region with explicit hysteresis thresholds.
    ///
    /// # Arguments
    /// - `on_threshold`: Energy must exceed this to become active
    /// - `off_threshold`: Energy must drop below this to become quiet
    ///
    /// # Panics
    /// Debug-asserts that off_threshold <= on_threshold
    pub fn with_hysteresis(
        name: impl Into<String>,
        range: Range<usize>,
        on_threshold: u32,
        off_threshold: u32,
    ) -> Self {
        debug_assert!(
            off_threshold <= on_threshold,
            "off_threshold ({}) must be <= on_threshold ({})",
            off_threshold,
            on_threshold
        );
        Self {
            name: name.into(),
            range,
            on_threshold,
            off_threshold,
            weight: 100,
        }
    }

    /// Set the hysteresis gap as percentage (0-100).
    ///
    /// off_threshold = on_threshold * (100 - gap) / 100
    pub fn with_gap(mut self, gap: u8) -> Self {
        let gap = gap.min(100);
        self.off_threshold = self.on_threshold * (100 - gap as u32) / 100;
        self
    }

    /// Set weight (100 = 1.0×, 150 = 1.5×)
    pub fn with_weight(mut self, weight: u8) -> Self {
        self.weight = weight;
        self
    }

    /// Get the hysteresis gap as percentage.
    pub fn hysteresis_gap(&self) -> u8 {
        if self.on_threshold > 0 {
            (100 - (self.off_threshold * 100 / self.on_threshold)) as u8
        } else {
            0
        }
    }
}

impl Default for TriggerConfig {
    fn default() -> Self {
        Self {
            regions: Vec::new(),
            convergence_threshold: 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_hysteresis() {
        // Default gap is 20%
        let region = MonitoredRegion::new("test", 0..32, 1000);
        assert_eq!(region.on_threshold, 1000);
        assert_eq!(region.off_threshold, 800); // 1000 * 80 / 100
        assert_eq!(region.hysteresis_gap(), 20);
    }

    #[test]
    fn test_custom_hysteresis() {
        let region = MonitoredRegion::with_hysteresis("test", 0..32, 1000, 700);
        assert_eq!(region.on_threshold, 1000);
        assert_eq!(region.off_threshold, 700);
        assert_eq!(region.hysteresis_gap(), 30);
    }

    #[test]
    fn test_with_gap() {
        let region = MonitoredRegion::new("test", 0..32, 1000).with_gap(30);
        assert_eq!(region.on_threshold, 1000);
        assert_eq!(region.off_threshold, 700); // 1000 * 70 / 100
    }

    #[test]
    fn test_with_weight() {
        let region = MonitoredRegion::new("test", 0..32, 1000).with_weight(150);
        assert_eq!(region.weight, 150);
    }
}
