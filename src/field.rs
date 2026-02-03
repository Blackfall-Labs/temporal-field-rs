//! Temporal Field - Signal-based ring buffer with pub/sub
//!
//! The brain does not poll - one spark cascades.
//!
//! This is THE temporal field: ring buffer + decay + observer events.
//! Writes trigger downstream processing automatically.
//!
//! ASTRO_004 compliant: No floats. Signals throughout.

use crate::config::FieldConfig;
use crate::observer::{FieldEvent, FieldObserver, MonitoredRegion, TriggerConfig};
use crate::vector::FieldVector;
use std::ops::Range;
use std::sync::Arc;
use ternsig::Signal;

/// The temporal field - ring buffer with decay and pub/sub events.
///
/// Every write and tick checks thresholds and fires events to observers.
/// The brain does not poll - one spark cascades.
///
/// See crate-level documentation for full architecture explanation and examples.
///
/// # Clone behavior
///
/// Cloning a TemporalField copies the field state (frames, config, triggers)
/// but NOT the observers. The clone starts with no subscribers.
pub struct TemporalField {
    /// Ring buffer of frames.
    frames: Vec<FieldVector>,

    /// Configuration.
    config: FieldConfig,

    /// Current write position.
    write_head: usize,

    /// Total ticks elapsed.
    tick_count: u64,

    /// Registered observers for pub/sub.
    observers: Vec<Arc<dyn FieldObserver>>,

    /// What triggers notifications.
    triggers: TriggerConfig,

    /// Previous active state per region (for edge detection).
    was_active: Vec<bool>,
}

impl TemporalField {
    /// Create a new temporal field.
    ///
    /// After creation, configure the field:
    /// 1. Add monitored regions with `monitor_region()`
    /// 2. Subscribe observers with `subscribe()`
    /// 3. Writers write with `write_region()`, readers receive events
    pub fn new(config: FieldConfig) -> Self {
        let frames = (0..config.frame_count)
            .map(|_| FieldVector::new(config.dims))
            .collect();

        Self {
            frames,
            config,
            write_head: 0,
            tick_count: 0,
            observers: Vec::new(),
            triggers: TriggerConfig::default(),
            was_active: Vec::new(),
        }
    }

    /// Add a monitored region after construction.
    pub fn monitor_region(&mut self, region: MonitoredRegion) {
        self.triggers.regions.push(region);
        self.was_active.push(false);
    }

    /// Set convergence threshold.
    pub fn set_convergence_threshold(&mut self, threshold: usize) {
        self.triggers.convergence_threshold = threshold;
    }

    // =========================================================================
    // PUB/SUB - The brain does not poll
    // =========================================================================

    /// Subscribe an observer to receive field events.
    pub fn subscribe(&mut self, observer: Arc<dyn FieldObserver>) {
        self.observers.push(observer);
    }

    /// Remove all observers.
    pub fn clear_observers(&mut self) {
        self.observers.clear();
    }

    /// Fire an event to all observers.
    fn fire(&self, event: FieldEvent) {
        for observer in &self.observers {
            observer.on_event(event.clone());
        }
    }

    /// Check regions and fire events for state changes.
    ///
    /// Uses hysteresis to prevent chattering:
    /// - To become active: energy must exceed on_threshold
    /// - To become quiet: energy must drop below off_threshold
    /// - Between thresholds: maintain previous state
    fn check_and_fire(&mut self) {
        if self.triggers.regions.is_empty() {
            return;
        }

        let mut active_regions = Vec::new();
        let mut total_energy: u32 = 0;

        for (i, region) in self.triggers.regions.iter().enumerate() {
            let energy = self.frames[self.write_head].range_energy(region.range.clone());
            let was = self.was_active.get(i).copied().unwrap_or(false);

            // Hysteresis logic:
            // - If already active, stay active until energy drops below off_threshold
            // - If not active, only become active if energy exceeds on_threshold
            let is_active = if was {
                // Already active - use lower threshold to leave
                energy >= region.off_threshold
            } else {
                // Not active - use higher threshold to enter
                energy > region.on_threshold
            };

            // Edge detection: became active (crossed on_threshold from below)
            if is_active && !was {
                self.fire(FieldEvent::RegionActive {
                    region: region.range.clone(),
                    energy,
                    threshold: region.on_threshold,
                });
            }

            // Edge detection: became quiet (dropped below off_threshold)
            if !is_active && was {
                self.fire(FieldEvent::RegionQuiet {
                    region: region.range.clone(),
                    energy,
                    threshold: region.off_threshold,
                });
            }

            // Track for convergence
            if is_active {
                active_regions.push(region.range.clone());
                // Weighted energy: energy × weight / 100 (since weight 100 = 1.0×)
                total_energy += (energy as u64 * region.weight as u64 / 100) as u32;
            }

            // Update state
            if i < self.was_active.len() {
                self.was_active[i] = is_active;
            }
        }

        // Check for convergence (multiple regions active)
        if active_regions.len() >= self.triggers.convergence_threshold {
            self.fire(FieldEvent::Convergence {
                active_regions,
                total_energy,
            });
        }
    }

    // =========================================================================
    // TIME ADVANCEMENT
    // =========================================================================

    /// Advance time by one tick - decay all frames, may fire RegionQuiet events.
    pub fn tick(&mut self) {
        self.tick_count += 1;
        for frame in &mut self.frames {
            frame.decay(self.config.retention);
        }
        self.check_and_fire();
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
    // WRITING - triggers event checks after mutation
    // =========================================================================

    /// Write Signals to a region of the current frame (additive) - may fire events.
    pub fn write_region(&mut self, signals: &[Signal], range: Range<usize>) {
        self.frames[self.write_head].add_to_range(signals, range);
        self.check_and_fire();
    }

    /// Set Signals in a region of the current frame (replace) - may fire events.
    pub fn set_region(&mut self, signals: &[Signal], range: Range<usize>) {
        self.frames[self.write_head].set_range(signals, range);
        self.check_and_fire();
    }

    /// Add a full vector to current frame - may fire events.
    pub fn write_full(&mut self, vector: &FieldVector) {
        self.frames[self.write_head].add(vector);
        self.check_and_fire();
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
    pub fn read_region(&self, range: Range<usize>) -> Vec<Signal> {
        self.frames[self.write_head].get_range(range)
    }

    /// Get energy in a region of current frame.
    pub fn region_energy(&self, range: Range<usize>) -> u32 {
        self.frames[self.write_head].range_energy(range)
    }

    /// Check if region is active (energy above threshold).
    pub fn region_active(&self, range: Range<usize>, threshold: u32) -> bool {
        self.region_energy(range) > threshold
    }

    /// Read the last N frames in chronological order (oldest first).
    pub fn read_window(&self, n: usize) -> Vec<&FieldVector> {
        let n = n.min(self.config.frame_count);
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let idx = (self.write_head + self.config.frame_count - n + i)
                % self.config.frame_count;
            result.push(&self.frames[idx]);
        }

        result
    }

    /// Get peak values in a region over the last N frames.
    /// Returns the frame with highest energy.
    pub fn region_peak(&self, range: Range<usize>, window: usize) -> Vec<Signal> {
        let frames = self.read_window(window);
        if frames.is_empty() {
            return vec![Signal::ZERO; range.len()];
        }

        let mut best_frame_idx = 0;
        let mut best_energy = 0u32;

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
    /// Returns averaged Signal values (via i16 arithmetic).
    pub fn region_mean(&self, range: Range<usize>, window: usize) -> Vec<Signal> {
        let frames = self.read_window(window);
        if frames.is_empty() {
            return vec![Signal::ZERO; range.len()];
        }

        let len = range.len();
        let mut sums: Vec<i32> = vec![0; len];

        for frame in &frames {
            for (i, idx) in range.clone().enumerate() {
                sums[i] += frame.get_i16(idx) as i32;
            }
        }

        let n = frames.len() as i32;
        sums.iter()
            .map(|&sum| Signal::from_signed_i32(sum / n))
            .collect()
    }

    // =========================================================================
    // METRICS
    // =========================================================================

    /// Get configuration.
    pub fn config(&self) -> &FieldConfig {
        &self.config
    }

    /// Get trigger configuration.
    pub fn triggers(&self) -> &TriggerConfig {
        &self.triggers
    }

    /// Get monitored regions.
    pub fn regions(&self) -> &[MonitoredRegion] {
        &self.triggers.regions
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

    /// Get maximum magnitude in field.
    pub fn max_magnitude(&self) -> u8 {
        self.frames
            .iter()
            .map(|f| f.max_magnitude())
            .max()
            .unwrap_or(0)
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
        self.was_active.fill(false);
    }

    /// Convert tick difference to milliseconds.
    pub fn ticks_to_ms(&self, ticks: u64) -> u32 {
        ((ticks as u64 * 1000) / self.config.tick_rate_hz as u64) as u32
    }

    /// Convert milliseconds to ticks.
    pub fn ms_to_ticks(&self, ms: u32) -> u64 {
        (ms as u64 * self.config.tick_rate_hz as u64) / 1000
    }
}

impl Clone for TemporalField {
    /// Clone the field state but NOT the observers.
    /// The clone starts with no subscribers.
    fn clone(&self) -> Self {
        Self {
            frames: self.frames.clone(),
            config: self.config.clone(),
            write_head: self.write_head,
            tick_count: self.tick_count,
            observers: Vec::new(), // Observers are not cloned
            triggers: self.triggers.clone(),
            was_active: self.was_active.clone(),
        }
    }
}

impl std::fmt::Debug for TemporalField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemporalField")
            .field("dims", &self.config.dims)
            .field("frame_count", &self.config.frame_count)
            .field("retention", &self.config.retention)
            .field("write_head", &self.write_head)
            .field("tick_count", &self.tick_count)
            .field("observers", &self.observers.len())
            .field("regions", &self.triggers.regions.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_new_field() {
        let config = FieldConfig::new(64, 10, 242); // 242 ≈ 0.95
        let field = TemporalField::new(config);

        assert_eq!(field.dims(), 64);
        assert_eq!(field.frame_count(), 10);
        assert_eq!(field.tick_count(), 0);
        assert_eq!(field.total_activity(), 0);
    }

    #[test]
    fn test_write_and_read_region() {
        let config = FieldConfig::new(128, 10, 242);
        let mut field = TemporalField::new(config);

        let signals = vec![Signal::positive(128); 32];
        field.write_region(&signals, 0..32);

        // Energy = 32 * 128^2 = 524288
        assert!(field.region_active(0..32, 1000));
        assert!(!field.region_active(32..64, 1000));
    }

    #[test]
    fn test_decay() {
        let config = FieldConfig::new(64, 10, 128); // 50% retention
        let mut field = TemporalField::new(config);

        let signals = vec![Signal::positive(200); 64];
        field.write_region(&signals, 0..64);
        let initial = field.region_energy(0..64);

        field.tick();
        let after_tick = field.region_energy(0..64);

        // After 50% decay, energy should be ~25% (magnitude halved, energy = mag^2)
        assert!(after_tick < initial / 2);
    }

    #[test]
    fn test_region_active_fires_event() {
        let config = FieldConfig::new(64, 10, 242);
        let mut field = TemporalField::new(config);

        // Configure: add monitored region
        // Threshold: 32 * 100^2 = 320000
        field.monitor_region(MonitoredRegion::new("test", 0..32, 100_000));

        // Subscribe reader
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();
        field.subscribe(Arc::new(crate::observer::FnObserver(move |event| {
            if matches!(event, FieldEvent::RegionActive { .. }) {
                count_clone.fetch_add(1, Ordering::SeqCst);
            }
        })));

        // Writer writes - fires event to reader (magnitude 128, energy = 32 * 16384 = 524288)
        let signals = vec![Signal::positive(128); 32];
        field.write_region(&signals, 0..32);

        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_convergence_fires() {
        let config = FieldConfig::new(128, 10, 242);
        let mut field = TemporalField::new(config);

        // Configure: add monitored regions (threshold = 50000)
        field.monitor_region(MonitoredRegion::new("a", 0..32, 50_000));
        field.monitor_region(MonitoredRegion::new("b", 32..64, 50_000));
        field.monitor_region(MonitoredRegion::new("c", 64..96, 50_000));
        field.set_convergence_threshold(2);

        let convergence_count = Arc::new(AtomicUsize::new(0));
        let cc = convergence_count.clone();

        field.subscribe(Arc::new(crate::observer::FnObserver(move |event| {
            if matches!(event, FieldEvent::Convergence { .. }) {
                cc.fetch_add(1, Ordering::SeqCst);
            }
        })));

        // Write to two regions - should trigger convergence
        let signals = vec![Signal::positive(128); 32];
        field.write_region(&signals, 0..32);
        field.write_region(&signals, 32..64);

        assert!(convergence_count.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let config = FieldConfig::new(64, 3, 255); // No decay
        let mut field = TemporalField::new(config);

        for i in 0..5 {
            field.clear_current();
            let signals = vec![Signal::positive(((i + 1) * 25) as u8); 64];
            field.write_region(&signals, 0..64);
            field.advance_write_head();
        }

        assert_eq!(field.write_head(), 2);
    }

    #[test]
    fn test_window_chronological() {
        let config = FieldConfig::new(64, 5, 255); // No decay
        let mut field = TemporalField::new(config);

        for i in 0..3 {
            field.clear_current();
            let signals = vec![Signal::positive(((i + 1) * 50) as u8); 1];
            field.write_region(&signals, 0..1);
            field.advance_write_head();
        }

        let window = field.read_window(3);
        assert_eq!(window.len(), 3);

        assert_eq!(window[0].get(0).magnitude, 50);
        assert_eq!(window[1].get(0).magnitude, 100);
        assert_eq!(window[2].get(0).magnitude, 150);
    }

    #[test]
    fn test_hysteresis_prevents_chattering() {
        // Use single dimension for simpler energy calculation
        // Energy = magnitude^2 for single dimension
        let config = FieldConfig::new(1, 10, 255); // No decay for clarity
        let mut field = TemporalField::new(config);

        // Region with explicit hysteresis:
        // on_threshold = 10000 (mag ~100), off_threshold = 2500 (mag ~50)
        field.monitor_region(MonitoredRegion::with_hysteresis("test", 0..1, 10000, 2500));

        let active_count = Arc::new(AtomicUsize::new(0));
        let quiet_count = Arc::new(AtomicUsize::new(0));
        let ac = active_count.clone();
        let qc = quiet_count.clone();

        field.subscribe(Arc::new(crate::observer::FnObserver(move |event| {
            match event {
                FieldEvent::RegionActive { .. } => {
                    ac.fetch_add(1, Ordering::SeqCst);
                }
                FieldEvent::RegionQuiet { .. } => {
                    qc.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
        })));

        // Write magnitude 120 → energy = 14400 (above on_threshold 10000) → should fire RegionActive
        field.set_region(&[Signal::positive(120)], 0..1);
        assert_eq!(active_count.load(Ordering::SeqCst), 1, "Should fire RegionActive");
        assert_eq!(quiet_count.load(Ordering::SeqCst), 0, "Should not fire RegionQuiet");

        // Write magnitude 70 → energy = 4900 (between thresholds: below on=10000 but above off=2500)
        // Should NOT fire any event due to hysteresis
        field.set_region(&[Signal::positive(70)], 0..1);
        assert_eq!(active_count.load(Ordering::SeqCst), 1, "Should not fire again (hysteresis)");
        assert_eq!(quiet_count.load(Ordering::SeqCst), 0, "Should stay active (hysteresis)");

        // Write magnitude 40 → energy = 1600 (below off_threshold 2500) → should fire RegionQuiet
        field.set_region(&[Signal::positive(40)], 0..1);
        assert_eq!(active_count.load(Ordering::SeqCst), 1, "Should not fire RegionActive");
        assert_eq!(quiet_count.load(Ordering::SeqCst), 1, "Should fire RegionQuiet");

        // Write magnitude 70 → energy = 4900 (above off=2500 but below on=10000)
        // Should NOT fire any event (need to exceed on_threshold to become active again)
        field.set_region(&[Signal::positive(70)], 0..1);
        assert_eq!(active_count.load(Ordering::SeqCst), 1, "Should not become active (hysteresis)");
        assert_eq!(quiet_count.load(Ordering::SeqCst), 1, "Should stay quiet");

        // Write magnitude 120 → energy = 14400 (above on_threshold 10000) → should fire RegionActive again
        field.set_region(&[Signal::positive(120)], 0..1);
        assert_eq!(active_count.load(Ordering::SeqCst), 2, "Should fire RegionActive again");
    }

    #[test]
    fn test_region_mean() {
        let config = FieldConfig::new(4, 5, 255); // No decay
        let mut field = TemporalField::new(config);

        // Write 3 frames with different values
        field.set_region(&[Signal::positive(60)], 0..1);
        field.advance_write_head();
        field.set_region(&[Signal::positive(120)], 0..1);
        field.advance_write_head();
        field.set_region(&[Signal::positive(180)], 0..1);
        field.advance_write_head();

        let mean = field.region_mean(0..1, 3);
        // (60 + 120 + 180) / 3 = 120
        assert_eq!(mean[0].magnitude, 120);
    }
}
