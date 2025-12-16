"""
HLX LC Codec - Latent Collapse Binary/Text Encoding
Deterministic 1:1 bijective encoding for HLX values.
"""

import struct
import hashlib
import math
import re
from typing import Any, Dict, List, Tuple, Union, Optional

from .errors import (
    E_FLOAT_SPECIAL, E_DEPTH_EXCEEDED,
    E_LC_PARSE, E_LC_DECODE, E_LC_ENCODE,
    E_FIELD_ORDER
)

LC_TAGS = {
    'NULL': 0x00, 'INT': 0x01, 'FLOAT': 0x02, 'TEXT': 0x03,
    'BYTES': 0x04, 'ARR_START': 0x05, 'ARR_END': 0x06,
    'OBJ_START': 0x07, 'OBJ_END': 0x08, 'HANDLE_REF': 0x09,
    'BOOL_TRUE': 0x0A, 'BOOL_FALSE': 0x0B,
}

TAG_NAMES = {v: k for k, v in LC_TAGS.items()}


class LCCodecError(Exception):
    pass

class LCEncodeError(LCCodecError):
    pass

class LCDecodeError(LCCodecError):
    pass


def encode_uleb128(value: int) -> bytes:
    if value < 0:
        raise LCEncodeError("ULEB128 cannot encode negative values")
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        result.append(byte)
        if value == 0:
            break
    return bytes(result)


def decode_uleb128(data: bytes, offset: int = 0) -> Tuple[int, int]:
    result, shift, size = 0, 0, 0
    while offset + size < len(data):
        byte = data[offset + size]
        size += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result, size


def encode_sleb128(value: int) -> bytes:
    result = bytearray()
    more = True
    while more:
        byte = value & 0x7F
        value >>= 7
        if (value == 0 and (byte & 0x40) == 0) or (value == -1 and (byte & 0x40) != 0):
            more = False
        else:
            byte |= 0x80
        result.append(byte)
    return bytes(result)


def decode_sleb128(data: bytes, offset: int = 0) -> Tuple[int, int]:
    result, shift, size = 0, 0, 0
    while offset + size < len(data):
        byte = data[offset + size]
        size += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if (byte & 0x80) == 0:
            if shift < 64 and (byte & 0x40):
                result |= -(1 << shift)
            break
    else:
        # Loop exited because no more data (not because of break)
        # This means the last byte had the continuation bit set
        if size > 0 and (data[offset + size - 1] & 0x80) != 0:
            raise LCDecodeError(f"Incomplete SLEB128 at offset {offset}: continuation bit set but no following byte")
    return result, size


def encode_float64_be(value: float) -> bytes:
    if math.isnan(value) or math.isinf(value):
        raise LCEncodeError(f"{E_FLOAT_SPECIAL}: NaN/Inf not allowed")
    # Normalize zero
    if value == 0.0:
        value = 0.0 # Ensures -0.0 becomes 0.0
    return struct.pack('>d', value)

def decode_float64_be(data: bytes, offset: int = 0) -> Tuple[float, int]:
    val = struct.unpack('>d', data[offset:offset + 8])[0]
    if math.isnan(val) or math.isinf(val):
         raise LCDecodeError(f"{E_FLOAT_SPECIAL}: NaN/Inf encountered during decode")
    return val, 8


class LCBParser:
    """
    CONTRACT_800: LC-B Binary Parser
    """
    def parse(self, data: bytes) -> Any:
        return LCBinaryDecoder(data).decode()

    def encode(self, value: Any) -> bytes:
        return LCBinaryEncoder().encode(value)


class LCBinaryEncoder:
    def __init__(self):
        self.buffer = bytearray()

    def encode(self, value: Any) -> bytes:
        self.buffer = bytearray()
        self._encode_value(value, 0)
        return bytes(self.buffer)

    def _encode_value(self, value: Any, depth: int):
        if depth > 64:
            raise LCEncodeError(f"{E_DEPTH_EXCEEDED}: Max recursion depth 64 exceeded")

        if value is None:
            self.buffer.append(LC_TAGS['NULL'])
        elif isinstance(value, bool):
            self.buffer.append(LC_TAGS['BOOL_TRUE'] if value else LC_TAGS['BOOL_FALSE'])
        elif isinstance(value, int):
            self.buffer.append(LC_TAGS['INT'])
            self.buffer.extend(encode_sleb128(value))
        elif isinstance(value, float):
            self.buffer.append(LC_TAGS['FLOAT'])
            self.buffer.extend(encode_float64_be(value))
        elif isinstance(value, str):
            if value.startswith('&h_'):
                self.buffer.append(LC_TAGS['HANDLE_REF'])
            else:
                self.buffer.append(LC_TAGS['TEXT'])
            encoded = value.encode('utf-8')
            self.buffer.extend(encode_uleb128(len(encoded)))
            self.buffer.extend(encoded)
        elif isinstance(value, (bytes, bytearray)):
            self.buffer.append(LC_TAGS['BYTES'])
            self.buffer.extend(encode_uleb128(len(value)))
            self.buffer.extend(value)
        elif isinstance(value, list):
            self.buffer.append(LC_TAGS['ARR_START'])
            self.buffer.extend(encode_uleb128(len(value)))
            for item in value:
                self._encode_value(item, depth + 1)
            self.buffer.append(LC_TAGS['ARR_END'])
        elif isinstance(value, dict):
            self.buffer.append(LC_TAGS['OBJ_START'])
            # INV-003: Sort keys with special handling for @N numeric fields
            # @0, @1, @2, @10 should sort numerically, not lexicographically
            def sort_key(k):
                if k.startswith('@') and k[1:].isdigit():
                    return (0, int(k[1:]))  # Numeric fields sort first, by number
                else:
                    return (1, k)  # Non-numeric fields sort second, lexicographically

            sorted_keys = sorted(value.keys(), key=sort_key)
            self.buffer.extend(encode_uleb128(len(sorted_keys)))
            for key in sorted_keys:
                if not isinstance(key, str):
                    raise LCEncodeError(f"Keys must be strings, got {type(key)}")
                key_bytes = key.encode('utf-8')
                self.buffer.extend(encode_uleb128(len(key_bytes)))
                self.buffer.extend(key_bytes)
                self._encode_value(value[key], depth + 1)
            self.buffer.append(LC_TAGS['OBJ_END'])
        else:
            raise LCEncodeError(f"Cannot encode type: {type(value)}")


class LCBinaryDecoder:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def decode(self) -> Any:
        return self._decode_value(0)

    def _read_byte(self) -> int:
        if self.offset >= len(self.data):
            raise LCDecodeError("Unexpected end of data")
        byte = self.data[self.offset]
        self.offset += 1
        return byte

    def _read_uleb128(self) -> int:
        value, size = decode_uleb128(self.data, self.offset)
        self.offset += size
        return value

    def _read_sleb128(self) -> int:
        value, size = decode_sleb128(self.data, self.offset)
        self.offset += size
        return value

    def _read_bytes(self, count: int) -> bytes:
        if self.offset + count > len(self.data):
            raise LCDecodeError("Unexpected end of data")
        result = self.data[self.offset:self.offset + count]
        self.offset += count
        return result

    def _decode_value(self, depth: int) -> Any:
        if depth > 64:
             raise LCDecodeError(f"{E_DEPTH_EXCEEDED}: Max recursion depth 64 exceeded")

        tag = self._read_byte()

        if tag == LC_TAGS['NULL']:
            return None
        elif tag == LC_TAGS['BOOL_TRUE']:
            return True
        elif tag == LC_TAGS['BOOL_FALSE']:
            return False
        elif tag == LC_TAGS['INT']:
            return self._read_sleb128()
        elif tag == LC_TAGS['FLOAT']:
            value, _ = decode_float64_be(self.data, self.offset)
            self.offset += 8
            return value
        elif tag == LC_TAGS['TEXT']:
            length = self._read_uleb128()
            return self._read_bytes(length).decode('utf-8')
        elif tag == LC_TAGS['BYTES']:
            length = self._read_uleb128()
            return self._read_bytes(length)
        elif tag == LC_TAGS['HANDLE_REF']:
            length = self._read_uleb128()
            return self._read_bytes(length).decode('utf-8')
        elif tag == LC_TAGS['ARR_START']:
            count = self._read_uleb128()
            result = [self._decode_value(depth + 1) for _ in range(count)]
            if self._read_byte() != LC_TAGS['ARR_END']:
                raise LCDecodeError("Expected ARR_END")
            return result
        elif tag == LC_TAGS['OBJ_START']:
            count = self._read_uleb128()
            result = {}
            prev_key = None
            for _ in range(count):
                key_length = self._read_uleb128()
                key = self._read_bytes(key_length).decode('utf-8')
                if prev_key is not None:
                    # INV-003: Keys must be sorted with numeric handling for @N fields
                    def sort_val(k):
                        if k.startswith('@') and k[1:].isdigit():
                            return (0, int(k[1:]))
                        else:
                            return (1, k)
                    if sort_val(key) <= sort_val(prev_key):
                        if key == prev_key:
                            raise LCDecodeError(f"{E_FIELD_ORDER}: Duplicate key {key}")
                        else:
                            raise LCDecodeError(f"{E_FIELD_ORDER}: Keys not sorted: {prev_key} >= {key}")
                prev_key = key
                result[key] = self._decode_value(depth + 1)
            if self._read_byte() != LC_TAGS['OBJ_END']:
                raise LCDecodeError("Expected OBJ_END")
            return result
        else:
            raise LCDecodeError(f"Unknown tag: 0x{tag:02x}")


class LCTParser:
    """
    CONTRACT_801: LC-T Text Parser
    Pedagogical format: [OBJ_START, FIELD_0, INT(123), OBJ_END]
    """
    def parse_text(self, text: str) -> Any:
        text = text.strip()
        if not text.startswith('[') or not text.endswith(']'):
            raise LCDecodeError("LC-T text must be enclosed in brackets []")

        content = text[1:-1]
        self.tokens = self._tokenize(content)
        self.idx = 0

        # Parse exactly one value from the stream
        value = self._parse_from_tokens()

        # Verify all tokens were consumed (INV-003 equivalent for text)
        if self.idx < len(self.tokens):
            raise LCDecodeError(f"Unexpected trailing tokens: {self.tokens[self.idx:]}")

        return value

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        current = []
        in_quote = False
        parens = 0
        
        for char in text:
            if char == '"':
                in_quote = not in_quote
                current.append(char)
            elif char == '(' and not in_quote:
                parens += 1
                current.append(char)
            elif char == ')' and not in_quote:
                parens -= 1
                current.append(char)
            elif char == ',' and not in_quote and parens == 0:
                token = "".join(current).strip()
                if token: tokens.append(token)
                current = []
            else:
                current.append(char)
        
        token = "".join(current).strip()
        if token: tokens.append(token)
        return tokens

    def _parse_from_tokens(self) -> Any:
        if self.idx >= len(self.tokens):
            raise LCDecodeError("Unexpected end of token stream")

        token = self.tokens[self.idx]
        self.idx += 1

        if token == 'NULL': return None
        if token == 'TRUE': return True
        if token == 'FALSE': return False

        if token.startswith('INT(') and token.endswith(')'):
            return int(token[4:-1])
        if token.startswith('FLOAT(') and token.endswith(')'):
            val = float(token[6:-1])
            if math.isnan(val) or math.isinf(val):
                raise LCDecodeError(f"{E_FLOAT_SPECIAL}: NaN/Inf not allowed in LC-T")
            return val
        if token.startswith('STRING(') and token.endswith(')'):
            content = token[7:-1]
            if content.startswith('"') and content.endswith('"'):
                return content[1:-1].replace('\"', '"')
            return content # Should be quoted usually
        if token.startswith('HANDLE(') and token.endswith(')'):
            return token[7:-1]
            
        if token == 'OBJ_START':
            obj = {}
            while True:
                if self.idx >= len(self.tokens):
                     raise LCDecodeError("Unclosed Object")
                
                peek = self.tokens[self.idx]
                if peek == 'OBJ_END':
                    self.idx += 1
                    break
                
                # Expect key
                self.idx += 1 # Consume key token
                key = self._parse_key(peek)
                
                # Expect value
                val = self._parse_from_tokens()
                obj[key] = val
            return obj
            
        if token == 'ARR_START':
            arr = []
            while True:
                if self.idx >= len(self.tokens):
                     raise LCDecodeError("Unclosed Array")
                
                peek = self.tokens[self.idx]
                if peek == 'ARR_END':
                    self.idx += 1
                    break
                
                val = self._parse_from_tokens()
                arr.append(val)
            return arr
            
        raise LCDecodeError(f"Unknown token: {token}")

    def _parse_key(self, token: str) -> str:
        if token.startswith('FIELD_'):
            return str(token[6:]) # FIELD_0 -> "0"
        if token.startswith('KEY(') and token.endswith(')'):
             content = token[4:-1]
             if content.startswith('"') and content.endswith('"'):
                 return content[1:-1].replace('\"', '"')
             return content
        raise LCDecodeError(f"Expected Key, got {token}")

    def to_text(self, value: Any) -> str:
        tokens = []
        self._tokenize_value(value, tokens)
        return "[" + ", ".join(tokens) + "]"

    def _tokenize_value(self, value: Any, tokens: List[str]):
        if value is None:
            tokens.append("NULL")
        elif isinstance(value, bool):
            tokens.append("TRUE" if value else "FALSE")
        elif isinstance(value, int):
            tokens.append(f"INT({value})")
        elif isinstance(value, float):
            tokens.append(f"FLOAT({value})")
        elif isinstance(value, str):
            if value.startswith('&h_'):
                tokens.append(f"HANDLE({value})")
            else:
                esc = value.replace('"', '\"')
                tokens.append(f'STRING("{esc}")')
        elif isinstance(value, (bytes, bytearray)):
             # Fallback for bytes if not spec'd, using BYTES([hex])
             tokens.append(f"BYTES([{value.hex()}])")
        elif isinstance(value, list):
            tokens.append("ARR_START")
            for item in value:
                self._tokenize_value(item, tokens)
            tokens.append("ARR_END")
        elif isinstance(value, dict):
            tokens.append("OBJ_START")
            for k in sorted(value.keys()):
                if k.isdigit():
                    tokens.append(f"FIELD_{k}")
                else:
                    esc = k.replace('"', '\"')
                    tokens.append(f'KEY("{esc}")')
                self._tokenize_value(value[k], tokens)
            tokens.append("OBJ_END")
        else:
             raise LCEncodeError(f"Cannot encode to LC-T: {type(value)}")


RUNIC_GLYPHS = {
    'COLLAPSE': '\U0001F71A', 'INT': '\u16C7', 'FLOAT': '\u16DE',
    'TEXT': '\u16A6', 'BYTES': '\u16B2', 'ARRAY': '\u16C8',
    'OBJECT': '\u16DF', 'HANDLE': '\u16B9', 'NULL': '\u16C9',
    'TRUE': '\u16CF', 'FALSE': '\u16A0',
}


def encode_runic(value: Any) -> str:
    if value is None:
        return RUNIC_GLYPHS['NULL']
    elif isinstance(value, bool):
        return RUNIC_GLYPHS['TRUE'] if value else RUNIC_GLYPHS['FALSE']
    elif isinstance(value, int):
        return f"{RUNIC_GLYPHS['INT']}{value}"
    elif isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise LCEncodeError(f"{E_FLOAT_SPECIAL}: NaN/Inf not allowed in runic encoding")
        return f"{RUNIC_GLYPHS['FLOAT']}{value}"
    elif isinstance(value, str):
        if value.startswith('&h_'):
            return f"{RUNIC_GLYPHS['HANDLE']}{value}"
        escaped = value.replace('\\', '\\\\').replace('"', '\"')
        return f'{RUNIC_GLYPHS["TEXT"]}"{escaped}"'
    elif isinstance(value, (bytes, bytearray)):
        return f"{RUNIC_GLYPHS['BYTES']}[{value.hex()}]"
    elif isinstance(value, list):
        items = [encode_runic(item) for item in value]
        return f"{RUNIC_GLYPHS['ARRAY']}[{', '.join(items)}]"
    elif isinstance(value, dict):
        pairs = [f'"{k}":{encode_runic(value[k])}' for k in sorted(value.keys())]
        return f"{RUNIC_GLYPHS['OBJECT']}{{{', '.join(pairs)}}}"
    else:
        raise LCEncodeError(f"Cannot encode to LC-T: {type(value)}")

# Alias for backward compatibility
encode_lct = encode_runic


def encode_lcb(value: Any) -> bytes:
    return LCBinaryEncoder().encode(value)

def decode_lcb(data: bytes) -> Any:
    return LCBinaryDecoder(data).decode()

def compute_hash(data: bytes) -> str:
    # Contract 802 specifies BLAKE2b-256 as primary
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def get_type_tag(value: Any) -> str:
    if value is None: return "null"
    if isinstance(value, bool): return "bool"
    if isinstance(value, int): return "int"
    if isinstance(value, float): return "float"
    if isinstance(value, str): return "str"
    if isinstance(value, (bytes, bytearray)): return "blob"
    if isinstance(value, list): return "list"
    if isinstance(value, dict): return "map"
    return "unknown"

def canonical_hash(value: Any) -> str:
    return compute_hash(encode_lcb(value))

def verify_bijection(value: Any) -> bool:
    encoded = encode_lcb(value)
    decoded = decode_lcb(encoded)
    return encode_lcb(decoded) == encoded

def wrap_contract(contract_id: int, value: Any) -> Dict:
    return {str(contract_id): {"@0": value}}

def unwrap_contract(wrapped: Dict) -> Tuple[int, Any]:
    if len(wrapped) != 1:
        raise LCDecodeError("Contract must have exactly one key")
    contract_id_str = list(wrapped.keys())[0]
    contract_id = int(contract_id_str)
    inner = wrapped[contract_id_str]
    if "@0" not in inner:
        raise LCDecodeError("Contract inner must have @0 field")
    return contract_id, inner["@0"]