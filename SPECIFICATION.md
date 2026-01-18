# Temporal Field Specification

**The brain does not poll - one spark cascades.**

## 0. Architecture: Field / Writer / Reader

The system separates into three roles:

### 0.1 The Field

The substrate - a ring buffer with decay. Holds:
- `frames: Vec<FieldVector>` - the ring buffer
- `write_head: usize` - current write position
- `tick_count: u64` - time counter
- `observers: Vec<Arc<dyn FieldObserver>>` - subscribed readers
- `triggers: TriggerConfig` - what fires events

### 0.2 Writers

Anything that calls write methods:
- Cochlea writes audio features to `0..64`
- Tokenizer writes text embeddings to `64..128`
- Motor mesh writes action patterns to `128..192`

Writers don't know about each other. They just write to their region.
Additive semantics allow multiple writers to contribute to the same frame.

### 0.3 Readers

FieldObservers that receive events when thresholds cross:

```rust
pub trait FieldObserver: Send + Sync {
    fn on_event(&self, event: FieldEvent);
}
```

Readers subscribe with `field.subscribe(observer)` and receive:
- `RegionActive` - a region crossed above threshold
- `RegionQuiet` - a region dropped below threshold
- `Convergence` - N+ regions simultaneously active

### 0.4 The Pattern

```rust
// 1. Create field
let mut field = TemporalField::new(config);

// 2. Configure what triggers events
field.monitor_region(MonitoredRegion::new("audio", 0..64, 0.1));
field.monitor_region(MonitoredRegion::new("text", 64..128, 0.1));
field.set_convergence_threshold(2);

// 3. Subscribe readers
field.subscribe(Arc::new(my_binding_detector));

// 4. Writers write - events fire automatically
field.write_region(&audio_features, 0..64);   // may fire RegionActive
field.write_region(&text_embedding, 64..128); // may fire Convergence
```

## 1. Floating Ternary Representation

### 1.1 Core Encoding

Each value in a FieldVector is stored as a signed byte representing intensity:

```
Storage:  i8 in range [-100, +100]
Semantic: f32 in range [-1.00, +1.00]

Conversion:
  store(f32) → (clamp(-1.0, 1.0) × 100).round() as i8
  load(i8)   → i8 as f32 / 100.0
```

### 1.2 Why "Floating Ternary"

The term captures the essence of the encoding:

- **Ternary polarity**: Values are negative, zero, or positive
- **Floating magnitude**: Intensity varies continuously within each polarity

This differs from:
- **Discrete ternary** `{-1, 0, +1}`: No intensity information
- **Floating point** `f32`: Unbounded, 4× memory, precision overkill

### 1.3 Precision

```
Resolution: 0.01 (1% of full scale)
Quantization levels: 201 (-100 to +100 inclusive)
Bits required: 8 (7.65 theoretical minimum)
```

Two decimal places is sufficient for neural activations. Higher precision would waste memory without improving cognitive behavior.

## 2. FieldVector Operations

### 2.1 Decay

```rust
fn decay(&mut self, retention: f32) {
    for v in &mut self.values {
        let current = *v as f32 / 100.0;
        let decayed = current * retention;
        *v = (decayed * 100.0).round() as i8;
    }
}
```

Decay preserves sign and shrinks magnitude toward zero. With `retention = 0.95`:

```
Tick 0:  1.00 → 1.00
Tick 1:  1.00 → 0.95
Tick 10: 1.00 → 0.60
Tick 50: 1.00 → 0.08
```

Half-life formula: `ticks = ln(0.5) / ln(retention)`

| Retention | Half-life (ticks) | At 100Hz |
|-----------|-------------------|----------|
| 0.99      | 69                | 690ms    |
| 0.995     | 138               | 1.38s    |
| 0.95      | 14                | 140ms    |
| 0.90      | 7                 | 70ms     |

### 2.2 Addition (Saturating)

```rust
fn add(&mut self, other: &FieldVector) {
    for i in 0..self.values.len() {
        let sum = (self.values[i] as i16) + (other.values[i] as i16);
        self.values[i] = sum.clamp(-100, 100) as i8;
    }
}
```

Multiple writers can contribute to the same vector. Saturation at ±1.0 prevents overflow and enforces bounded activations.

### 2.3 Energy Calculation

```rust
fn range_energy(&self, range: Range<usize>) -> f32 {
    range.map(|i| {
        let v = self.get(i);
        v * v
    }).sum()
}
```

Energy is the sum of squared values in a region. This measures activity regardless of polarity.

## 3. TemporalField

### 3.1 Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    TEMPORAL FIELD SUBSTRATE                  │
│                                                              │
│  Frame 0   Frame 1   Frame 2   ...   Frame N-1              │
│  ┌──────┐  ┌──────┐  ┌──────┐       ┌──────┐               │
│  │ dim0 │  │ dim0 │  │ dim0 │       │ dim0 │               │
│  │ dim1 │  │ dim1 │  │ dim1 │       │ dim1 │               │
│  │ ...  │  │ ...  │  │ ...  │       │ ...  │               │
│  │ dimD │  │ dimD │  │ dimD │       │ dimD │               │
│  └──────┘  └──────┘  └──────┘       └──────┘               │
│     ↑                                                        │
│  write_head (advances on advance_write_head())              │
│                                                              │
│  tick() decays ALL frames simultaneously                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Configuration

```rust
pub struct FieldConfig {
    dims: usize,        // Vector dimensions per frame
    frame_count: usize, // Ring buffer depth
    retention: f32,     // Decay factor per tick [0.0, 1.0]
    tick_rate_hz: u32,  // For time calculations
}
```

Memory footprint: `dims × frame_count` bytes

### 3.3 Time Model

Time advances via explicit `tick()` calls. Each tick:

1. Increments `tick_count`
2. Applies decay to ALL frames (not just current)

This means:
- Values written 10 ticks ago have decayed 10×
- No timestamps needed - time is encoded in magnitude
- Decay is continuous, not tied to write head

### 3.4 Write Semantics

**Additive writes** (default):
```rust
field.write_region(&values, 0..32);  // Adds to existing
```

**Replacement writes**:
```rust
field.set_region(&values, 0..32);    // Overwrites existing
```

Additive semantics allow multiple sources to contribute without coordination:
- Cochlea writes audio features
- Tokenizer writes text embeddings
- Motor mesh writes action patterns
- All accumulate in the same frame

### 3.5 Region Partitioning

Dimensions can be logically partitioned into regions:

```
dims 0-63:   Audio features
dims 64-127: Visual features
dims 128-191: Text embeddings
dims 192-255: Action patterns
```

The substrate doesn't enforce regions - it's a convention. Each writer knows its region and writes there.

## 4. Temporal Queries

### 4.1 Current Frame

```rust
let current = field.read_current();      // &FieldVector
let values = field.read_region(0..32);   // Vec<f32>
let energy = field.region_energy(0..32); // f32
let active = field.region_active(0..32, 0.1); // bool
```

### 4.2 Historical Window

```rust
let window = field.read_window(5);  // Vec<&FieldVector>, oldest first
```

Returns the last N frames in chronological order. Useful for:
- Detecting co-occurrence patterns
- Computing temporal averages
- Finding peak activity

### 4.3 Aggregation

```rust
// Best frame in window (highest energy)
let peak = field.region_peak(0..32, 5);

// Mean across window
let mean = field.region_mean(0..32, 5);
```

## 5. Design Philosophy

### 5.1 Time as Space

The ring buffer makes time a spatial dimension:
- Frame N-5 = "5 ticks ago"
- Indexable, queryable, visible
- No hidden state mystery

### 5.2 Physics Over Learning

Decay is physics, not learned:
- Recent patterns are stronger
- Old patterns fade automatically
- No gating decisions required

### 5.3 Memory Efficiency

i8 storage is deliberate:
- Bounded activations by construction
- Predictable memory footprint
- Fast integer operations

### 5.4 Composability

The same substrate serves multiple purposes:
- Sensory binding (BindingField)
- Cognitive integration (ConvergenceField)
- Social salience (FocusField)

Configuration differs; mechanism is identical.

## 6. Pub/Sub Events

### 6.1 MonitoredRegion with Hysteresis

A region to watch for activity, with hysteresis to prevent threshold chattering:

```rust
pub struct MonitoredRegion {
    pub name: String,         // For debugging
    pub range: Range<usize>,  // Which dimensions
    pub on_threshold: f32,    // Energy threshold to become active (higher)
    pub off_threshold: f32,   // Energy threshold to become quiet (lower)
    pub weight: f32,          // Contribution to convergence energy
}
```

**Why hysteresis?** Crisp thresholds can chatter when values hover near the boundary
(especially with decay + additive writes). Two thresholds provide stable transitions:

```
Energy axis:
                    off_threshold    on_threshold
         ──────────────┼─────────────────┼──────────────→
         quiet         │   hysteresis    │   active
                       │     zone        │
                       │                 │
         ←─ stay quiet ┤                 ├─ become active
                       │                 │
         become quiet ─┤                 ├─ stay active →
```

- To **enter active**: energy must exceed `on_threshold`
- To **leave active**: energy must drop below `off_threshold`
- **Between thresholds**: previous state is maintained

Default hysteresis gap is 20%: `off_threshold = on_threshold * 0.8`

```rust
// Automatic hysteresis (20% gap)
MonitoredRegion::new("audio", 0..64, 0.1)
// on_threshold = 0.1, off_threshold = 0.08

// Explicit hysteresis
MonitoredRegion::with_hysteresis("audio", 0..64, 0.1, 0.05)
// on_threshold = 0.1, off_threshold = 0.05

// Custom gap
MonitoredRegion::new("audio", 0..64, 0.1).with_gap(0.3)
// on_threshold = 0.1, off_threshold = 0.07
```

### 6.2 TriggerConfig

Configuration for what fires events:

```rust
pub struct TriggerConfig {
    pub regions: Vec<MonitoredRegion>,
    pub convergence_threshold: usize, // How many regions for Convergence
}
```

### 6.3 FieldEvent

Events fired to observers:

```rust
pub enum FieldEvent {
    /// A region crossed above on_threshold (edge: inactive → active)
    RegionActive {
        region: Range<usize>,
        energy: f32,
        threshold: f32,  // The on_threshold that was crossed
    },

    /// A region dropped below off_threshold (edge: active → inactive)
    RegionQuiet {
        region: Range<usize>,
        energy: f32,
        threshold: f32,  // The off_threshold that was crossed
    },

    /// Multiple regions are simultaneously active
    Convergence {
        active_regions: Vec<Range<usize>>,
        total_energy: f32,
    },
}
```

### 6.4 Edge Detection with Hysteresis

Events fire on **transitions**, not continuous states:
- `RegionActive` fires once when energy crosses above `on_threshold`
- `RegionQuiet` fires once when energy drops below `off_threshold`
- No events when energy is between thresholds (hysteresis zone)
- No repeated events while region stays active/quiet

This prevents event storms and threshold chattering, matching biological spike semantics.
The hysteresis zone acts as a refractory period without explicit refractory state.

### 6.5 When Events Fire

Events are checked after:
- `write_region()` - writer may push region above threshold
- `set_region()` - writer may change region state
- `write_full()` - writer may affect multiple regions
- `tick()` - decay may drop regions below threshold

## 7. Origin

This substrate was extracted from a larger cognitive architecture project to enforce a single API plane for temporal field operations across multiple applications.

The pub/sub pattern was added to implement the principle: **The brain does not poll - one spark cascades.**
