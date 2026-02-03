//! Temporal Field - Ring buffer substrate for cognitive architectures
//!
//! The brain does not poll - one spark cascades.
//!
//! ASTRO_004 compliant: No floats. Signal (polarity + magnitude) throughout.
//!
//! # Core Types
//!
//! - **Signal**: Universal 2-byte neural signal (polarity: i8, magnitude: u8)
//! - **TemporalField**: Ring buffer with decay and pub/sub events
//!
//! # Architecture: Field / Writer / Reader
//!
//! The system separates into three roles:
//!
//! 1. **Field** - The substrate: ring buffer with decay
//! 2. **Writers** - Anything that writes to regions (cochlea, tokenizer, motor, etc.)
//! 3. **Readers** - FieldObservers that receive events when thresholds cross
//!
//! Multiple writers can write to the same field (additive semantics).
//! Multiple readers can subscribe to the same field (pub/sub).
//! Writes automatically fire events to all readers when thresholds cross.
//!
//! # Core Concepts
//!
//! - **Ring buffer**: Fixed memory, oldest frames auto-evicted
//! - **Decay per tick**: Time encoded in values, not metadata (retention: u8, 255 = 1.0)
//! - **Regions**: Spatial partitioning for multi-channel integration
//! - **Pub/sub**: Writes fire events to observers automatically
//! - **Signal**: Compact neural representation (s = polarity × magnitude)
//!
//! # Integer Conventions
//!
//! | Parameter | Type | Scale | Examples |
//! |-----------|------|-------|----------|
//! | retention | u8 | 255 = 1.0 | 242 ≈ 0.95, 230 ≈ 0.90, 128 = 0.50 |
//! | threshold | u32 | sum of magnitude² | 524288 = 32 dims × 128² |
//! | weight | u8 | 100 = 1.0× | 150 = 1.5×, 80 = 0.8× |
//! | energy | u32 | Σ(magnitude²) | max = 64 × 255² = 4,161,600 |
//!
//! # Example: Multimodal Binding Detection
//!
//! This example shows the pattern used for concept grounding - detecting when
//! audio and text patterns co-occur (e.g., hearing "door" while seeing the word).
//!
//! ```rust
//! use temporal_field::{TemporalField, FieldConfig, FieldEvent, MonitoredRegion, FnObserver};
//! use ternsig::Signal;
//! use std::sync::Arc;
//!
//! // Region definitions (like SensoryField's ModalityRegions)
//! const AUDIO_REGION: std::ops::Range<usize> = 0..64;
//! const TEXT_REGION: std::ops::Range<usize> = 64..128;
//!
//! // 1. Create the Field (the substrate)
//! // 128 dims, 50 frames, retention 242 (≈0.95)
//! let config = FieldConfig::new(128, 50, 242);
//! let mut field = TemporalField::new(config);
//!
//! // 2. Configure monitored regions (what triggers events)
//! // Threshold: energy needed to activate (e.g., 100_000 = ~50 dims at mag 45)
//! field.monitor_region(MonitoredRegion::new("audio", AUDIO_REGION, 100_000));
//! field.monitor_region(MonitoredRegion::new("text", TEXT_REGION, 100_000));
//! field.set_convergence_threshold(2); // Fire when 2+ regions active
//!
//! // 3. Subscribe Reader (binding detector)
//! field.subscribe(Arc::new(FnObserver(|event| {
//!     match event {
//!         FieldEvent::Convergence { active_regions, total_energy } => {
//!             // Binding opportunity! Audio + text co-occurred
//!             println!("BINDING: {} regions, energy={}", active_regions.len(), total_energy);
//!         }
//!         FieldEvent::RegionActive { region, energy, .. } => {
//!             println!("Region {:?} activated with energy {}", region, energy);
//!         }
//!         _ => {}
//!     }
//! })));
//!
//! // 4. Writers write to their regions using Signals
//! // Cochlea writes audio features (magnitude 128 = moderate activation)
//! let audio_features = vec![Signal::positive(128); 64];
//! field.write_region(&audio_features, AUDIO_REGION);
//!
//! // Tokenizer writes text embedding
//! let text_embedding = vec![Signal::positive(100); 64];
//! field.write_region(&text_embedding, TEXT_REGION);
//! // ^ This triggers Convergence event because both regions are now active
//!
//! // 5. Time advances - decay happens automatically
//! field.tick(); // All magnitudes decay by retention factor
//! ```
//!
//! # Key Insight
//!
//! The field doesn't know what audio or text means. It just knows that patterns
//! co-occurred within a temporal window. Meaning emerges from the binding.

mod config;
mod field;
mod observer;
mod vector;

pub use config::FieldConfig;
pub use field::TemporalField;
pub use observer::{FieldEvent, FieldObserver, FnObserver, MonitoredRegion, TriggerConfig};
pub use vector::FieldVector;

// Signal: Re-export from ternsig (the authoritative source)
pub use ternsig::Signal;

// Deprecated alias for v1 compatibility
#[deprecated(since = "0.4.0", note = "Use Signal instead")]
pub type TernarySignal = Signal;
