//! Observer pattern for temporal fields - brain-style pub/sub
//!
//! When a write crosses a threshold, observers are notified automatically.
//! No polling required - sparks propagate.
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
        energy: f32,
        /// The on_threshold that was crossed
        threshold: f32,
    },
    /// A region went quiet (energy dropped below off_threshold)
    RegionQuiet {
        region: Range<usize>,
        energy: f32,
        /// The off_threshold that was crossed
        threshold: f32,
    },
    /// Multiple regions active simultaneously (binding opportunity)
    Convergence {
        active_regions: Vec<Range<usize>>,
        total_energy: f32,
    },
    /// Peak detected in a region (local maximum)
    Peak {
        region: Range<usize>,
        energy: f32,
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

/// A region being monitored for activity with hysteresis thresholds.
///
/// ## Hysteresis
///
/// Two thresholds control state transitions to prevent chattering:
/// - `on_threshold`: Energy must exceed this to become active
/// - `off_threshold`: Energy must drop below this to become quiet
///
/// When energy is between the thresholds, the previous state is maintained.
#[derive(Clone, Debug)]
pub struct MonitoredRegion {
    /// Name for identification
    pub name: String,
    /// Dimension range
    pub range: Range<usize>,
    /// Energy threshold to enter active state (higher threshold)
    pub on_threshold: f32,
    /// Energy threshold to leave active state (lower threshold)
    pub off_threshold: f32,
    /// Weight for convergence calculation
    pub weight: f32,
}

/// Default hysteresis gap as a fraction of on_threshold.
/// off_threshold = on_threshold * (1.0 - HYSTERESIS_GAP)
pub const DEFAULT_HYSTERESIS_GAP: f32 = 0.2;

impl MonitoredRegion {
    /// Create a new monitored region with automatic hysteresis.
    ///
    /// The off_threshold is set to 80% of the on_threshold by default,
    /// providing a 20% hysteresis gap to prevent chattering.
    pub fn new(name: impl Into<String>, range: Range<usize>, threshold: f32) -> Self {
        Self {
            name: name.into(),
            range,
            on_threshold: threshold,
            off_threshold: threshold * (1.0 - DEFAULT_HYSTERESIS_GAP),
            weight: 1.0,
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
        on_threshold: f32,
        off_threshold: f32,
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
            weight: 1.0,
        }
    }

    /// Set the hysteresis gap as a fraction (0.0 to 1.0).
    ///
    /// off_threshold = on_threshold * (1.0 - gap)
    pub fn with_gap(mut self, gap: f32) -> Self {
        self.off_threshold = self.on_threshold * (1.0 - gap.clamp(0.0, 1.0));
        self
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Get the hysteresis gap as a fraction.
    pub fn hysteresis_gap(&self) -> f32 {
        if self.on_threshold > 0.0 {
            1.0 - (self.off_threshold / self.on_threshold)
        } else {
            0.0
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
