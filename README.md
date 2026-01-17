# Temporal Field Substrate

A shared ring buffer with decay for cognitive architectures.

## Core Concept

**Time is a dimension, not a loop.**

Traditional RNNs process time sequentially, accumulating hidden state through recurrence. This creates gradient flow problems and opaque internal dynamics.

Temporal Fields make time **spatial**:
- Ring buffer = time as an indexable dimension
- `read_window(n)` = query last n frames
- Decay = natural forgetting without explicit gating
- No gradients, no backprop through time

## The Floating Ternary Foundation

Each dimension encodes **direction + intensity** using signed bytes:

```
Range: {-1.00 ... 0.00 ... +1.00}
Storage: i8 (-100 to +100) for 2 decimal precision
```

This is NOT discrete ternary `{-1, 0, +1}`. It's **floating-point ternary** - the polarity indicates direction, the magnitude indicates strength.

Why i8 instead of f32?
- 1 byte per dimension vs 4 bytes
- 256 dims × 50 frames = 12.5 KB (not 50 KB)
- Integer operations for decay/add
- Quantization forces bounded activations

## Applications

| Field Type | Purpose | Configuration |
|------------|---------|---------------|
| **BindingField** | Sensory pattern integration | 256 dims, 500ms window |
| **ConvergenceField** | Mesh output integration | 64 dims, 100ms window |
| **FocusField** | Interlocutor connection salience | 8 dims, 1.4s half-life |

All share the same substrate - only configuration differs.

## Key Properties

- **Ring buffer**: Fixed memory, oldest frames auto-evicted
- **Decay per tick**: Time encoded in values, not metadata
- **Regions**: Spatial partitioning for multi-channel integration
- **Additive writes**: Multiple writers can contribute to same frame

## Usage

```rust
use temporal_field::{TemporalFieldSubstrate, SubstrateConfig};

// Create a 64-dimensional field with 10 frames, 95% retention
let config = SubstrateConfig::new(64, 10, 0.95);
let mut field = TemporalFieldSubstrate::new(config);

// Write to a region (additive)
field.write_region(&vec![0.5; 32], 0..32);

// Advance time (decay happens)
field.tick();

// Check region activity
if field.region_active(0..32, 0.1) {
    println!("Region is active!");
}

// Read temporal window (oldest first)
let window = field.read_window(5);
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Ring buffer | O(1) memory, natural history eviction |
| Additive writes | Multiple writers, no coordination needed |
| Energy-based activity | Robust to sparse patterns |
| Configurable decay | Domain-specific tuning |
| Dimension regions | Modality separation without separate buffers |

## License

MIT OR Apache-2.0
