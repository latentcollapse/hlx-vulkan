# HLX Backend Comprehensive Audit Report
## Execution Date: 2025-12-16

---

## Executive Summary

**Status**: CRITICAL ISSUES FOUND - PRODUCTION NOT READY

The backend has **6 major logic gaps** that violate fundamental axioms and invariants. These issues will cause determinism failures, contract violations, and data corruption in production.

**Severity Breakdown**:
- **CRITICAL**: 4 issues (breaks axioms/invariants)
- **IMPORTANT**: 2 issues (data corruption/spec violations)
- **MINOR**: 3 issues (performance/edge cases)

---

## CRITICAL ISSUES

### ISSUE #1: Field Ordering Uses Lexicographic Sort (INV-003 Violation)
**Severity**: CRITICAL
**Location**: `hlx_runtime/lc_codec.py:162` - `LCBinaryEncoder._encode_value()`
**Impact**: Breaks field ordering invariant for numeric field names

**Problem**:
```python
sorted_keys = sorted(value.keys())  # Line 162
```
This sorts keys lexicographically (alphabetically), not numerically.

**Example**:
```python
dict: {"@0": "a", "@1": "b", "@2": "c", "@10": "d"}
Expected order: @0 < @1 < @2 < @10 (numeric)
Actual order:   @0 < @1 < @10 < @2 (lexicographic)
```

**Test Results**:
```
Decoded order: ['@0', '@1', '@10', '@2']
Numeric order: ['@0', '@1', '@2', '@10']
Lexical order: ['@0', '@1', '@10', '@2']  ← MATCHES ACTUAL (WRONG!)
```

**Violated Contract**: INV-003 (FIELD_ORDER)
**Violated Axiom**: Implicit ordering assumption in CONTRACT_805

**Fix Required**:
```python
# Parse numeric field indices
sorted_keys = sorted(value.keys(), key=lambda k: (
    int(k[1:]) if k.startswith('@') and k[1:].isdigit() else float('inf'),
    k
))
```

---

### ISSUE #2: SLEB128 Decoder Accepts Incomplete Data
**Severity**: CRITICAL
**Location**: `hlx_runtime/lc_codec.py:79-90` - `decode_sleb128()`
**Impact**: Malformed encoded data silently succeeds, corrupting values

**Problem**:
The decoder accepts bytes with continuation bit set (0x80) but no following byte.

**Example**:
```python
incomplete = bytes([0xFF])  # Continuation bit set, but no next byte
value, size = decode_sleb128(incomplete, 0)
# Returns: (127, 1)  ← SHOULD ERROR!
```

**Test Results**:
```
✗ FAIL: Decoded incomplete SLEB128 as 127
```

**Root Cause**:
```python
while offset + size < len(data):  # Loop exits when no more bytes
    byte = data[offset + size]
    size += 1
    # ... check (byte & 0x80) == 0 to break
# If last byte has 0x80 set, loop just exits without checking
```

**Fix Required**:
```python
def decode_sleb128(data: bytes, offset: int = 0) -> Tuple[int, int]:
    result, shift, size = 0, 0, 0
    while offset + size < len(data):
        byte = data[offset + size]
        size += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if (byte & 0x80) == 0:
            break
    else:  # Loop exited without break = incomplete
        raise LCDecodeError(f"Incomplete SLEB128 at offset {offset}")
    return result, size
```

---

### ISSUE #3: NaN/Inf Slip-Through LCTParser
**Severity**: CRITICAL
**Location**: `hlx_runtime/lc_codec.py:331-332` - `LCTParser._parse_from_tokens()`
**Impact**: E_FLOAT_SPECIAL constraint violated, determinism broken

**Problem**:
LCTParser accepts `FLOAT(nan)` and `FLOAT(inf)` without validation.

**Test Results**:
```
✗ FAIL: [FLOAT(nan)] -> nan
✗ FAIL: [FLOAT(inf)] -> inf
```

**Code Issue**:
```python
if token.startswith('FLOAT(') and token.endswith(')'):
    return float(token[6:-1])  # No validation! Python float() accepts 'nan'/'inf'
```

**Fix Required**:
```python
if token.startswith('FLOAT(') and token.endswith(')'):
    val = float(token[6:-1])
    if math.isnan(val) or math.isinf(val):
        raise LCDecodeError(f"{E_FLOAT_SPECIAL}: NaN/Inf not allowed in LC-T")
    return val
```

---

### ISSUE #4: encode_runic() Doesn't Validate NaN/Inf
**Severity**: CRITICAL
**Location**: `hlx_runtime/lc_codec.py:438-461` - `encode_runic()`
**Impact**: E_FLOAT_SPECIAL constraint violated, runic encoding produces invalid data

**Problem**:
`encode_runic(float('nan'))` returns `'ᛞnan'` instead of raising error.

**Test Results**:
```
✗ FAIL: encode_runic(nan) -> ᛞnan
```

**Code Issue**:
```python
elif isinstance(value, float):
    return f"{RUNIC_GLYPHS['FLOAT']}{value}"  # No validation!
```

**Fix Required**:
```python
elif isinstance(value, float):
    if math.isnan(value) or math.isinf(value):
        raise LCEncodeError(f"{E_FLOAT_SPECIAL}: NaN/Inf not allowed")
    return f"{RUNIC_GLYPHS['FLOAT']}{value}"
```

---

## IMPORTANT ISSUES

### ISSUE #5: LCTParser Allows Trailing Data (Multi-Value Streams)
**Severity**: IMPORTANT
**Location**: `hlx_runtime/lc_codec.py:283-289` - `LCTParser.parse_text()`
**Impact**: Accepts malformed input, silently ignores data, breaks round-trip invariant

**Problem**:
Parser accepts multiple values but only returns first, silently ignoring rest.

**Example**:
```python
lct.parse_text("[INT(42), INT(999)]")  # Should error or parse both
# Returns: 42  ← Silently ignored INT(999)!
```

**Test Results**:
```
✗ FAIL: Accepted multi-value: 42
```

**Code Issue**:
```python
value = self._parse_from_tokens()
if self.idx < len(self.tokens):
    pass  # TODO comment suggests this is unresolved
return value
```

**Fix Required**:
```python
value = self._parse_from_tokens()
if self.idx < len(self.tokens):
    raise LCDecodeError(f"Unexpected trailing tokens: {self.tokens[self.idx:]}")
return value
```

---

### ISSUE #6: Zero Normalization Works, But Implicit in Code
**Severity**: MINOR
**Location**: `hlx_runtime/lc_codec.py:97-98` - `encode_float64_be()`
**Status**: WORKING (but code is confusing)

**Observation**:
```python
# Normalize zero
if value == 0.0:
    value = 0.0 # Ensures -0.0 becomes 0.0
```

This actually works (Python treats -0.0 == 0.0 as True, and assigns positive zero). But the comment is misleading. The code should be:

```python
# Normalize -0.0 to 0.0 for determinism
if value == 0.0:
    value = abs(value) * 0.0  # Better: explicit
# Or even better:
value = 0.0 if value == 0.0 else value
```

---

## MINOR ISSUES

### ISSUE #7: Drop Guard Doesn't Cleanup (By Design)
**Severity**: MINOR
**Location**: `src/context.rs:929-941` - `Drop impl`
**Status**: INTENTIONAL (but risky)

**Observation**:
The Drop guard explicitly doesn't call cleanup() because "Vulkan driver may already be torn down."

**Risk**: If user forgets `ctx.cleanup()`, resources leak silently (only warning logged).

**Recommendation**: Add `atexit` handler in Python to ensure cleanup:
```python
import atexit
ctx = VulkanContext()
atexit.register(ctx.cleanup)
```

---

### ISSUE #8: LCT Integer Parsing Doesn't Validate Range
**Severity**: MINOR
**Location**: `hlx_runtime/lc_codec.py:329-330`

**Problem**:
```python
if token.startswith('INT(') and token.endswith(')'):
    return int(token[4:-1])  # No range check, accepts arbitrarily large integers
```

**Observation**: Python ints are arbitrary precision, so this is OK for LCT. But for cross-language interchange, should validate against i64 range.

---

### ISSUE #9: ULEB128 Doesn't Validate Max Size
**Severity**: MINOR
**Location**: `hlx_runtime/lc_codec.py:38-50` - `encode_uleb128()`

**Observation**: No maximum size check. Very large values could encode to massive byte sequences. Should add:
```python
if value > 0xFFFFFFFFFFFFFFFF:  # Or whatever limit
    raise LCEncodeError("ULEB128 value exceeds maximum")
```

---

## SUMMARY TABLE

| # | Issue | Severity | Type | Fix Effort | Risk |
|---|-------|----------|------|-----------|------|
| 1 | Field ordering (lexical vs numeric) | CRITICAL | Logic | Medium | HIGH - breaks INV-003 |
| 2 | SLEB128 incomplete data | CRITICAL | Logic | Low | HIGH - data corruption |
| 3 | NaN/Inf in LCTParser | CRITICAL | Validation | Low | HIGH - axiom violation |
| 4 | NaN/Inf in encode_runic | CRITICAL | Validation | Low | HIGH - axiom violation |
| 5 | LCTParser trailing data | IMPORTANT | Logic | Low | MEDIUM - silently fails |
| 6 | Zero normalization (code clarity) | MINOR | Documentation | Low | LOW - works correctly |
| 7 | Drop guard incomplete | MINOR | Resource mgmt | Low | LOW - warning only |
| 8 | LCT int range validation | MINOR | Validation | Low | LOW - Python ints OK |
| 9 | ULEB128 size limit | MINOR | Validation | Low | LOW - edge case |

---

## RECOMMENDED FIXES (Priority Order)

### Phase 1: CRITICAL (Must fix before production)
1. Fix ISSUE #1: Field ordering (numeric sort for @0, @1, etc.)
2. Fix ISSUE #2: SLEB128 incomplete data detection
3. Fix ISSUE #3: NaN/Inf in LCTParser
4. Fix ISSUE #4: NaN/Inf in encode_runic

### Phase 2: IMPORTANT (Should fix before 1.0)
5. Fix ISSUE #5: LCTParser trailing data rejection

### Phase 3: NICE-TO-HAVE (Future)
6. Clarify ISSUE #6: Zero normalization comment
7. Add ISSUE #7: Python atexit cleanup handler
8. Add ISSUE #8: LCT integer range validation
9. Add ISSUE #9: ULEB128 size limit

---

## HEAVYWEIGHT TOOLS FOR NEXT PHASE

Based on audit findings, recommend building:

1. **Determinism Verifier Tool** - Detect determinism violations
   - Runs encode/decode 1000x, verifies bit-identical output
   - Identifies non-deterministic code paths
   - Integrates with CI/CD

2. **Contract Validator Tool** - Deep field ordering validation
   - Validates @0 < @1 < @2 < @3... strictly
   - Tests numeric vs string keys
   - Generates test cases automatically

3. **Codec Fuzz Tester** - Finds edge cases in LC-B/LC-T
   - Random input generation
   - Crash detection
   - Coverage reporting

4. **Cross-Language Codec Tester** - Verify Python↔Rust bijection
   - Rust encodes → Python decodes
   - Python encodes → Rust decodes
   - Byte-for-byte comparison

---

## NEXT STEPS

1. ✓ Run audit (DONE)
2. → Fix CRITICAL issues (IN PROGRESS)
3. → Run extended test suite
4. → Build heavyweight tools
5. → Full regression test
6. → Ready for production

---

**Report Generated By**: Claude Code Haiku
**Execution Time**: ~30 minutes
**Code Reviewed**: ~3,000 LOC (Python + Rust)
**Tests Run**: 6 critical test cases
**Issues Found**: 9 (4 CRITICAL, 2 IMPORTANT, 3 MINOR)
