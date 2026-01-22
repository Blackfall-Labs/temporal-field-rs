//! Temporal Field - Ring buffer substrate for cognitive architectures
//!
//! The brain does not poll - one spark cascades.
//!
//! # Core Types
//!
//! - **TernarySignal**: Universal 2-byte neural signal (polarity + magnitude)
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
//! - **Decay per tick**: Time encoded in values, not metadata
//! - **Regions**: Spatial partitioning for multi-channel integration
//! - **Pub/sub**: Writes fire events to observers automatically
//! - **TernarySignal**: Compact neural representation (polarity + magnitude)
//!
//! # Evolution
//!
//! FloatingTernary → Temporal Binding Fields → Temporal Fields
//!
//! One system that evolved. Every write triggers downstream processing.
//!
//! # Example: Multimodal Binding Detection
//!
//! This example shows the pattern used for concept grounding - detecting when
//! audio and text patterns co-occur (e.g., hearing "door" while seeing the word).
//!
//! ```rust
//! use temporal_field::{TemporalField, FieldConfig, FieldEvent, MonitoredRegion, FnObserver};
//! use std::sync::Arc;
//!
//! // Region definitions (like SensoryField's ModalityRegions)
//! const AUDIO_REGION: std::ops::Range<usize> = 0..64;
//! const TEXT_REGION: std::ops::Range<usize> = 64..128;
//!
//! // 1. Create the Field (the substrate)
//! let config = FieldConfig::new(128, 50, 0.95); // 128 dims, 50 frames, 0.95 retention
//! let mut field = TemporalField::new(config);
//!
//! // 2. Configure monitored regions (what triggers events)
//! field.monitor_region(MonitoredRegion::new("audio", AUDIO_REGION, 0.1));
//! field.monitor_region(MonitoredRegion::new("text", TEXT_REGION, 0.1));
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
//! // 4. Writers write to their regions
//! // Cochlea writes audio features
//! let audio_features = vec![0.5; 64];
//! field.write_region(&audio_features, AUDIO_REGION);
//!
//! // Tokenizer writes text embedding
//! let text_embedding = vec![0.3; 64];
//! field.write_region(&text_embedding, TEXT_REGION);
//! // ^ This triggers Convergence event because both regions are now active
//!
//! // 5. Time advances - decay happens automatically
//! field.tick(); // All values decay by retention factor
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

// TernarySignal: Re-export from ternsig (the authoritative source)
pub use ternsig::TernarySignal;
