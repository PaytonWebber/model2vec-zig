//! Latin accent folding, standing in for the NFD-decompose-and-drop-marks
//! step of HuggingFace's BertNormalizer. Zig's std has no Unicode
//! normalization, so this folds the precomposed Latin-1 Supplement and Latin
//! Extended-A letters directly to their ASCII base and drops freestanding
//! combining marks (U+0300..U+036F). Letters that have no canonical
//! decomposition (æ, ø, ß, đ, ł, ...) pass through unchanged, exactly as NFD
//! leaves them.
//!
//! Scripts outside Latin are passed through unfolded; uppercase Greek or
//! Cyrillic input may tokenize to [UNK] where the reference implementation
//! lowercases it. English and Latin-script text matches the reference.

/// Fold one codepoint: lowercases ASCII and accented Latin letters, folds the
/// accent away, and returns 0 for codepoints that should be dropped
/// (combining marks). Everything else passes through.
pub fn fold(cp: u21) u21 {
    if (cp < 0x80) {
        return if (cp >= 'A' and cp <= 'Z') cp + 32 else cp;
    }
    if (cp >= 0x300 and cp <= 0x36F) return 0; // combining marks: drop
    return switch (cp) {
        // Latin-1 Supplement
        0xC0...0xC5, 0xE0...0xE5 => 'a',
        0xC7, 0xE7 => 'c',
        0xC8...0xCB, 0xE8...0xEB => 'e',
        0xCC...0xCF, 0xEC...0xEF => 'i',
        0xD1, 0xF1 => 'n',
        0xD2...0xD6, 0xF2...0xF6 => 'o',
        0xD9...0xDC, 0xF9...0xFC => 'u',
        0xDD, 0xFD, 0xFF => 'y',
        // Latin Extended-A, decomposable letters only
        0x100...0x105 => 'a',
        0x106...0x10D => 'c',
        0x10E, 0x10F => 'd',
        0x112...0x11B => 'e',
        0x11C...0x123 => 'g',
        0x124, 0x125 => 'h',
        0x128...0x130 => 'i',
        0x134, 0x135 => 'j',
        0x136, 0x137 => 'k',
        0x139...0x13E => 'l',
        0x143...0x148 => 'n',
        0x14C...0x151 => 'o',
        0x154...0x159 => 'r',
        0x15A...0x161 => 's',
        0x162...0x165 => 't',
        0x168...0x173 => 'u',
        0x174, 0x175 => 'w',
        0x176...0x178 => 'y',
        0x179...0x17E => 'z',
        else => cp,
    };
}

const std = @import("std");
const testing = std.testing;

test "fold lowercases and strips Latin accents" {
    try testing.expectEqual(@as(u21, 'a'), fold('A'));
    try testing.expectEqual(@as(u21, 'e'), fold('é'));
    try testing.expectEqual(@as(u21, 'e'), fold('É'));
    try testing.expectEqual(@as(u21, 'i'), fold('ï'));
    try testing.expectEqual(@as(u21, 'c'), fold('ç'));
    try testing.expectEqual(@as(u21, 's'), fold('š'));
    try testing.expectEqual(@as(u21, 0), fold(0x0301)); // combining acute
}

test "fold leaves non-decomposable letters alone" {
    try testing.expectEqual(@as(u21, 'ß'), fold('ß'));
    try testing.expectEqual(@as(u21, 'æ'), fold('æ'));
    try testing.expectEqual(@as(u21, 'ø'), fold('ø'));
    try testing.expectEqual(@as(u21, 'ł'), fold('ł'));
    try testing.expectEqual(@as(u21, 0x4E2D), fold(0x4E2D)); // CJK passes through
}
