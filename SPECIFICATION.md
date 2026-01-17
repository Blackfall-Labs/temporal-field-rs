# Temporal Field Substrate Specification

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

## 6. Origin

This substrate was extracted from a larger cognitive architecture project to enforce a single API plane for temporal field operations across multiple applications.
