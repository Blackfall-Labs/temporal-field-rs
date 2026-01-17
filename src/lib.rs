//! Temporal Field Substrate
//!
//! A shared ring buffer with decay for cognitive architectures.
//!
//! # Core Concept
//!
//! Temporal fields are the foundation for:
//! - **BindingField**: Sensory pattern integration
//! - **ConvergenceField**: Mesh output integration
//! - **FocusField**: Interlocutor connection salience
//!
//! All share the same core mechanism:
//! - **Ring buffer**: Fixed memory, oldest frames auto-evicted
//! - **Decay per tick**: Time encoded in values, not metadata
//! - **Regions**: Spatial partitioning for multi-channel integration
//! - **Additive writes**: Multiple writers can contribute to same frame
//!
//! # Example
//!
//! ```rust
//! use temporal_field::{TemporalField, FieldConfig};
//!
//! let config = FieldConfig::new(64, 10, 0.95);
//! let mut field = TemporalField::new(config);
//!
//! // Write to a region
//! field.write_region(&vec![0.5; 32], 0..32);
//!
//! // Advance time (decay happens)
//! field.tick();
//!
//! // Check region activity
//! assert!(field.region_active(0..32, 0.1));
//! ```

mod config;
mod field;
mod vector;

pub use config::FieldConfig;
pub use field::TemporalField;
pub use vector::FieldVector;
