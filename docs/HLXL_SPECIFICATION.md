# HLXL Specification - ASCII High-Level Language

## Overview

**HLXL** (HLX Language - Lite) is the human-readable ASCII form of HLX. It's designed to be:
- **Easy to learn** - Familiar syntax (Python-like)
- **Easy to write** - ASCII keyboard, no special glyphs
- **Easy to read** - Clear, self-documenting
- **Fully deterministic** - Reproducible across all hardware

---

## Lexical Structure

### Tokens

HLXL is composed of the following token types:

| Token | Pattern | Example |
|-------|---------|---------|
| **Number** | `-?[0-9]+(\.[0-9]+)?` | `42`, `-3.14`, `0` |
| **String** | `"..."` | `"hello"`, `"café"` |
| **Identifier** | `[a-zA-Z_][a-zA-Z0-9_]*` | `x`, `name`, `_private` |
| **Keyword** | `let`, `true`, `false`, `null` | — |
| **Operator** | `+`, `-`, `*`, `/`, `==`, etc. | — |
| **Bracket** | `[`, `]`, `{`, `}`, `(`, `)` | — |
| **Punctuation** | `:`, `,` | — |

### Reserved Words

```
let true false null and or not
print type collapse resolve snapshot transaction
```

### Comments

⚠️ **Comments are NOT supported** in HLXL for determinism reasons.

Rationale: Comments would add non-semantic information that must be stripped during parsing, introducing a source of non-determinism.

---

## Literals

### Null

```hlxl
null
```
Represents absence of value.

### Booleans

```hlxl
true
false
```

### Integers

```hlxl
0
42
-17
999999
2147483647
9223372036854775807    // 2^63 - 1 (max int64)
```

**Constraints:**
- Range: -2^63 to 2^63-1
- No underscores (use print for readability)
- No hex/octal/binary literals (write in decimal)

### Floats

```hlxl
3.14
-0.5
0.0
1e-6
1.5e10
```

**Constraints:**
- IEEE 754 format
- **NaN and Infinity NOT allowed** (raise `E_FLOAT_SPECIAL`)
- `-0.0` normalized to `0.0`
- Decimal point required (not `1.` or `.5`)

### Strings

```hlxl
"hello"
"Hello, World!"
"café"              // UTF-8 supported
"日本語"             // Full Unicode
""                  // Empty string
```

**String features:**
- UTF-8 encoding
- NFC normalization (composed form)
- Escape sequences: `\n` (newline), `\t` (tab), `\"` (quote), `\\` (backslash)
- Trailing whitespace trimmed for determinism

**Example:**
```hlxl
let greeting = "Hello\nWorld"    // Two lines
let quoted = "She said \"Hi\""    // Escaped quote
let path = "C:\\Users\\name"      // Escaped backslash
```

### Arrays

```hlxl
[]                      // Empty array
[1, 2, 3]              // Integer array
[1, "two", 3.0]        // Mixed types
[[1, 2], [3, 4]]       // Nested arrays
[null, true, "text"]   // Mixed types
```

**Features:**
- Heterogeneous (any types)
- Ordered (preserves sequence)
- Indexable: `arr[0]` gets first element
- No length limit (except memory)

**Syntax:**
```
[ element1 , element2 , ... , elementN ]
```

### Objects

```hlxl
{}                          // Empty object
{x: 1}                      // Single field
{a: 1, b: 2, c: 3}         // Multiple fields
{name: "Alice", age: 30}    // String keys
{nested: {inner: 42}}       // Nested objects
```

**Features:**
- Key-value pairs
- Keys are automatically **sorted lexicographically** (determinism)
- Values can be any type
- No duplicate keys allowed

**Syntax:**
```
{ key1: value1 , key2: value2 , ... }
```

**Key constraints:**
- Keys must be identifiers (no spaces, quotes, or special chars)
- Keys are case-sensitive
- Keys are sorted after parsing

**Example:**
```hlxl
let person = {z: 1, a: 2, m: 3}
// Automatically becomes: {a: 2, m: 3, z: 1}
```

---

## Variables & Binding

### Variable Declaration

```hlxl
let x = 42              // Bind 42 to x
let name = "Alice"      // Bind string
let arr = [1, 2, 3]     // Bind array
```

**Syntax:**
```
let identifier = expression
```

**Semantics:**
- Creates variable in current scope
- Variable is immediately available after binding
- Later bindings shadow earlier ones
- All bindings are returned (last expression is result)

**Example:**
```hlxl
let x = 10
let y = 20
let z = x + y
// z is 30, that's the return value
```

### Variable Access

```hlxl
let x = 42
print(x)        // Outputs: 42
```

**Scope:**
- Variables are in function scope
- New scope created for each function call
- Outer scope not accessible from inner scope

---

## Operators

### Arithmetic

```hlxl
10 + 5          // 15 (addition)
10 - 5          // 5 (subtraction)
10 * 5          // 50 (multiplication)
10 / 5          // 2.0 (floating-point division)
```

**Type rules:**
- `int + int → int`
- `int + float → float`
- `float + float → float`
- Division always returns float

**Examples:**
```hlxl
let a = 10 + 5 * 2     // 20 (multiplication first)
let b = (10 + 5) * 2   // 30 (parentheses first)
let c = 10 / 3         // 3.333... (float)
```

### Comparison

```hlxl
10 == 10        // true
10 != 5         // true
10 < 5          // false
10 <= 10        // true
10 > 5          // true
10 >= 10        // true
```

**Type rules:**
- Same types compared by value
- Different types: false (no coercion)
- Strings compared lexicographically

**Examples:**
```hlxl
"hello" == "hello"      // true
"hello" == "Hello"      // false (case-sensitive)
[1, 2] == [1, 2]       // true
{a: 1} == {a: 1}       // true
```

### Logical

```hlxl
true and false          // false
true or false           // true
not true                // false
```

**Short-circuit evaluation:**
```hlxl
true or error()         // Doesn't call error()
false and expensive()   // Doesn't call expensive()
```

### String Concatenation

```hlxl
"Hello" + " " + "World"     // "Hello World"
"Value: " + 42              // Type error (no coercion)
```

⚠️ String concatenation only works with `+` operator on strings. No automatic type conversion.

### Array/Object Access

```hlxl
let arr = [1, 2, 3]
arr[0]              // 1 (first element)
arr[2]              // 3 (third element)
arr[5]              // Error (out of bounds)

let obj = {a: 1, b: 2}
obj[a]              // Error (use dot notation or literals)
```

**Syntax:**
```
expression [ index ]
```

**Constraints:**
- Index must be integer literal or variable
- Index must be in bounds (0 to length-1)

---

## Operator Precedence

From highest to lowest:

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `[...]` (array access) | Left |
| 2 | `*`, `/` | Left |
| 3 | `+`, `-` | Left |
| 4 | `==`, `!=`, `<`, `>`, `<=`, `>=` | Left |
| 5 | `and` | Left |
| 6 | `or` | Left |
| 7 | `not` | Right |

**Examples:**
```hlxl
1 + 2 * 3           // 7 (multiply first)
true or false and false  // true (and binds tighter)
not false or true   // true (not binds tightest)
```

---

## Built-in Functions

### print(value)

Output a value to stdout.

```hlxl
print(42)
print("Hello")
print([1, 2, 3])
print({a: 1, b: 2})
```

**Behavior:**
- Outputs value in HLXL syntax
- Objects with keys sorted
- Returns the input value
- No newline added (use `"\n"` in string)

### type(value)

Get type of value as string.

```hlxl
type(42)            // "integer"
type("hello")       // "string"
type([1, 2])        // "array"
type({a: 1})        // "object"
type(null)          // "null"
type(true)          // "boolean"
type(3.14)          // "float"
```

**Returns:**
```
"null" | "boolean" | "integer" | "float" | "string" | "array" | "object" | "contract"
```

---

## Advanced Features

### Contracts (Type-Tagged Values)

Contracts allow schema-validated structures:

```hlxl
let user = @14 {
    @0: "alice",
    @1: 30,
    @2: true
}
```

**Syntax:**
```
@ contract_id { field: value, ... }
```

Contract 14 defines a user structure with:
- Field @0: string (name)
- Field @1: integer (age)
- Field @2: boolean (active)

**Validation:** Contract system validates at encode/decode time.

See [CONTRACT_SYSTEM.md](CONTRACT_SYSTEM.md) for details.

### Content-Addressed Storage (CAS)

Store values with content-based addressing:

```hlxl
let handle = store({name: "Alice", age: 30})
// handle is something like: &h_abc123def456...

let retrieved = retrieve(handle)
// retrieved is {age: 30, name: "Alice"}
```

**Functions:**
- `store(value)` → `&h_SHA256_HASH`
- `retrieve(handle)` → `value`
- `exists(handle)` → `boolean`

---

## Scoping & Environment

### Variable Scope

```hlxl
let x = 10
let y = 20
let z = x + y      // 30 (x and y available)

let inner_result =
    let a = 5
    a + x          // Error: x not in this scope
```

⚠️ **Note:** Each binding creates new scope. Outer variables not visible to inner bindings.

### Execution Order

```hlxl
let a = 1
let b = 2          // Uses a from above
let c = b + 1      // Uses b from above

print(c)           // 3
```

Sequential execution: variables available after binding.

---

## Type Coercion Rules

**NO automatic type coercion.** Errors on type mismatch:

```hlxl
"5" + 3             // Error: string concat needs string
5 == "5"            // false (different types, not error)
[1] + [2]           // Error: + not defined for arrays
```

**Exception:** Comparison with `==` and `!=` returns false for different types (no error).

---

## Error Handling

Errors are deterministic. Same input always produces same error:

| Error | Cause |
|-------|-------|
| `E_PARSE_ERROR` | Syntax error |
| `E_TYPE_ERROR` | Type mismatch in operation |
| `E_FLOAT_SPECIAL` | NaN or Infinity |
| `E_DEPTH_EXCEEDED` | Nesting >64 levels |
| `E_FIELD_ORDER` | Object keys not sorted |
| `E_HANDLE_NOT_FOUND` | CAS handle invalid |

**Example:**
```hlxl
let x = null
x + 5               // Error: E_TYPE_ERROR (null not addable)

let y = float('inf')    // Error: E_FLOAT_SPECIAL
```

---

## Complete Examples

### Example 1: Simple Arithmetic

```hlxl
let principal = 1000
let rate = 0.05
let time = 3

let interest = principal * rate * time
let total = principal + interest

print(total)        // 1150
```

### Example 2: Array Processing

```hlxl
let numbers = [1, 2, 3, 4, 5]
let first = numbers[0]
let last = numbers[4]
let sum = first + last

print(sum)          // 6
```

### Example 3: Data Structure

```hlxl
let student = {
    name: "Charlie",
    grade: 95,
    courses: ["Math", "Physics", "Chemistry"]
}

print(student)
// {courses: ["Chemistry", "Math", "Physics"], grade: 95, name: "Charlie"}
```

### Example 4: Conditional-like Logic (via functions)

```hlxl
let age = 25
let is_adult = age >= 18

print(is_adult)     // true
```

### Example 5: Contract Usage

```hlxl
let config = @100 {
    @0: "production",
    @1: true,
    @2: 8080
}

print(config)       // @100 {...}
```

---

## Determinism Guarantees in HLXL

### Guaranteed Properties

1. **Same code, same output** - Parse result deterministic
2. **No hidden state** - All operations visible
3. **No randomness** - No random() function
4. **No I/O** - print() is deterministic output only
5. **No concurrency** - Single-threaded execution

### What's NOT Deterministic

- **Floating-point operations may vary slightly** - IEEE 754 is deterministic per operation
- **Order of operations** - Depends on operator precedence (fixed)

### Best Practices

```hlxl
// ✓ Deterministic
let x = 10 + 20
let y = x * 2

// ✗ Non-deterministic (would need seed parameter)
// let r = random()

// ✓ Deterministic (consistent precision)
let pi = 3.14159265359
let area = pi * 100

// ✗ Non-deterministic (system-dependent)
// let timestamp = now()
```

---

## Runtime Behavior

### Execution Model

1. **Parse** - Convert text to AST
2. **Evaluate** - Execute statements sequentially
3. **Return** - Return value of last expression

### Memory Model

- **Values** - Immutable after creation
- **Variables** - Mutable references
- **Scope** - Function-local
- **Lifetime** - Until function returns

### Error Propagation

```hlxl
let x = 10
let y = x / 0       // Error: E_DIV_BY_ZERO
let z = y + 5       // Never executed
```

On error, execution stops and error is returned to caller.

---

## Next Steps

- **Learning examples:** See [../examples/](../examples/)
- **Running HLXL:** Use `HLXLRuntime` from runtime module
- **Converting to other forms:** See [LANGUAGE_SPECIFICATION.md](LANGUAGE_SPECIFICATION.md)
- **Type system details:** See [TYPE_SYSTEM.md](TYPE_SYSTEM.md)

---

**Version:** 1.1.0
**Status:** Ready for outside validation
**Testing:** 100+ test cases passing
