//! BERT-style WordPiece tokenizer, covering what the potion family of
//! model2vec models actually uses: BertNormalizer (clean text, lowercase,
//! strip accents, space out CJK) -> BertPreTokenizer (split on whitespace,
//! isolate punctuation) -> WordPiece with a `##` continuation prefix.
//!
//! This is not a general tokenizers.json engine. Models whose tokenizer is
//! BPE or Unigram are rejected at load.

const std = @import("std");
const accents = @import("accents.zig");

pub const Error = error{
    BadTokenizerJson,
    UnsupportedTokenizer,
} || std.mem.Allocator.Error;

pub const Tokenizer = struct {
    /// Owns the vocab strings; the map borrows from it.
    arena: std.heap.ArenaAllocator,
    vocab: std.StringHashMapUnmanaged(u32),
    unk_id: u32,
    max_chars_per_word: usize,
    /// Median byte length of vocab tokens; callers use it to bound input
    /// length in characters before tokenizing, as the reference does.
    median_token_len: usize,

    pub fn initFromJson(gpa: std.mem.Allocator, bytes: []const u8) Error!Tokenizer {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        const root = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch
            return error.BadTokenizerJson;
        if (root != .object) return error.BadTokenizerJson;
        const model = root.object.get("model") orelse return error.BadTokenizerJson;
        if (model != .object) return error.BadTokenizerJson;

        const kind = model.object.get("type") orelse return error.BadTokenizerJson;
        if (kind != .string or !std.mem.eql(u8, kind.string, "WordPiece")) {
            return error.UnsupportedTokenizer;
        }

        const vocab_val = model.object.get("vocab") orelse return error.BadTokenizerJson;
        if (vocab_val != .object) return error.BadTokenizerJson;

        var vocab: std.StringHashMapUnmanaged(u32) = .empty;
        try vocab.ensureTotalCapacity(a, @intCast(vocab_val.object.count()));

        var lens = try std.ArrayList(usize).initCapacity(a, vocab_val.object.count());
        var it = vocab_val.object.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* != .integer) return error.BadTokenizerJson;
            // Keys borrow from the parse arena, which this struct owns.
            vocab.putAssumeCapacity(entry.key_ptr.*, @intCast(entry.value_ptr.integer));
            lens.appendAssumeCapacity(entry.key_ptr.len);
        }
        std.mem.sort(usize, lens.items, {}, std.sort.asc(usize));
        const median = if (lens.items.len > 0) lens.items[lens.items.len / 2] else 1;

        const unk_name = blk: {
            const u = model.object.get("unk_token") orelse break :blk "[UNK]";
            break :blk if (u == .string) u.string else "[UNK]";
        };
        const unk_id = vocab.get(unk_name) orelse return error.BadTokenizerJson;

        const max_chars = blk: {
            const m = model.object.get("max_input_chars_per_word") orelse break :blk 100;
            break :blk if (m == .integer and m.integer > 0) @as(usize, @intCast(m.integer)) else 100;
        };

        return .{
            .arena = arena,
            .vocab = vocab,
            .unk_id = unk_id,
            .max_chars_per_word = max_chars,
            .median_token_len = median,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Tokenize `text` and append token ids to `out`. `scratch` is used for
    /// the normalized copy of the text; an arena is the natural fit.
    /// Does not add special tokens, matching model2vec inference.
    pub fn encode(self: *const Tokenizer, scratch: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        const normalized = try normalize(scratch, text);
        defer scratch.free(normalized);

        var words = std.mem.tokenizeScalar(u8, normalized, ' ');
        while (words.next()) |chunk| {
            // BertPreTokenizer: punctuation separates and becomes its own word.
            var start: usize = 0;
            var i: usize = 0;
            while (i < chunk.len) {
                const len = std.unicode.utf8ByteSequenceLength(chunk[i]) catch 1;
                const cp = std.unicode.utf8Decode(chunk[i .. i + len]) catch {
                    i += len;
                    continue;
                };
                if (isPunct(cp)) {
                    if (i > start) try self.wordPiece(scratch, chunk[start..i], out);
                    try self.wordPiece(scratch, chunk[i .. i + len], out);
                    start = i + len;
                }
                i += len;
            }
            if (start < chunk.len) try self.wordPiece(scratch, chunk[start..], out);
        }
    }

    /// Greedy longest-match-first WordPiece on a single word. A word with any
    /// unmatchable remainder becomes one [UNK], per the reference.
    fn wordPiece(self: *const Tokenizer, scratch: std.mem.Allocator, word: []const u8, out: *std.ArrayList(u32)) !void {
        const char_count = std.unicode.utf8CountCodepoints(word) catch word.len;
        if (char_count > self.max_chars_per_word) {
            try out.append(scratch, self.unk_id);
            return;
        }

        var pieces: std.ArrayList(u32) = .empty;
        defer pieces.deinit(scratch);
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(scratch);

        var start: usize = 0;
        while (start < word.len) {
            var end = word.len;
            const found: ?u32 = while (end > start) : (end = prevCharBoundary(word, end)) {
                buf.clearRetainingCapacity();
                if (start > 0) try buf.appendSlice(scratch, "##");
                try buf.appendSlice(scratch, word[start..end]);
                if (self.vocab.get(buf.items)) |id| break id;
            } else null;

            const id = found orelse {
                try out.append(scratch, self.unk_id);
                return;
            };
            try pieces.append(scratch, id);
            start = end;
        }
        try out.appendSlice(scratch, pieces.items);
    }
};

/// BertNormalizer: drop control characters, fold case and accents, surround
/// CJK ideographs with spaces, and map all whitespace to a single space.
fn normalize(scratch: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(scratch);
    try out.ensureTotalCapacity(scratch, text.len);

    var i: usize = 0;
    while (i < text.len) {
        const len = std.unicode.utf8ByteSequenceLength(text[i]) catch {
            i += 1; // invalid byte: drop, like clean_text drops garbage
            continue;
        };
        if (i + len > text.len) break;
        const cp = std.unicode.utf8Decode(text[i .. i + len]) catch {
            i += len;
            continue;
        };
        i += len;

        if (cp == 0 or cp == 0xFFFD) continue;
        if (isWhitespace(cp)) {
            try out.append(scratch, ' ');
            continue;
        }
        if (isControl(cp)) continue;

        const folded = accents.fold(cp);
        if (folded == 0) continue; // combining mark

        if (isCjk(folded)) {
            try out.append(scratch, ' ');
            try appendCp(scratch, &out, folded);
            try out.append(scratch, ' ');
        } else {
            try appendCp(scratch, &out, folded);
        }
    }
    return out.toOwnedSlice(scratch);
}

fn appendCp(scratch: std.mem.Allocator, out: *std.ArrayList(u8), cp: u21) !void {
    var buf: [4]u8 = undefined;
    const n = std.unicode.utf8Encode(cp, &buf) catch return;
    try out.appendSlice(scratch, buf[0..n]);
}

fn prevCharBoundary(s: []const u8, idx: usize) usize {
    var j = idx - 1;
    while (j > 0 and (s[j] & 0xC0) == 0x80) j -= 1;
    return j;
}

fn isWhitespace(cp: u21) bool {
    return switch (cp) {
        ' ', '\t', '\n', '\r' => true,
        0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

fn isControl(cp: u21) bool {
    if (cp < 0x20) return true; // \t \n \r already handled as whitespace
    return switch (cp) {
        0x7F...0x9F => true,
        0x200B...0x200F, 0x202A...0x202E, 0xFEFF => true, // format characters
        else => false,
    };
}

/// ASCII punctuation plus the common typographic marks. The reference treats
/// every Unicode P* category char as punctuation; this covers what shows up
/// in real prose and code. Anything missed stays inside its word and worst
/// case tokenizes to [UNK].
fn isPunct(cp: u21) bool {
    return switch (cp) {
        '!'...'/', ':'...'@', '['...'`', '{'...'~' => true,
        0x2010...0x2027 => true, // hyphens, dashes, quotes, daggers, ellipsis
        0x2030...0x205E => true, // permille, primes, brackets
        0xA1, 0xBF, 0xAB, 0xBB => true, // inverted marks, guillemets
        else => false,
    };
}

fn isCjk(cp: u21) bool {
    return switch (cp) {
        0x4E00...0x9FFF, 0x3400...0x4DBF => true,
        0xF900...0xFAFF => true,
        0x20000...0x2A6DF, 0x2A700...0x2B73F, 0x2B740...0x2B81F, 0x2B820...0x2CEAF => true,
        else => false,
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

fn testTokenizer(a: std.mem.Allocator) !Tokenizer {
    // Hand-built vocab exercising the continuation prefix and punctuation.
    return Tokenizer.initFromJson(a,
        \\{"model":{"type":"WordPiece","unk_token":"[UNK]","max_input_chars_per_word":100,
        \\"vocab":{"[UNK]":0,"hello":1,"world":2,"!":3,"zig":4,"##gy":5,"un":6,"##related":7,",":8,"the":9}}}
    );
}

fn encodeToSlice(t: *const Tokenizer, a: std.mem.Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    try t.encode(a, text, &out);
    return out.items;
}

test "encode lowercases, splits punctuation, and matches continuations" {
    var t = try testTokenizer(testing.allocator);
    defer t.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    try testing.expectEqualSlices(u32, &.{ 1, 8, 2, 3 }, try encodeToSlice(&t, a, "Hello, World!"));
    try testing.expectEqualSlices(u32, &.{ 4, 5 }, try encodeToSlice(&t, a, "ziggy"));
    try testing.expectEqualSlices(u32, &.{ 6, 7 }, try encodeToSlice(&t, a, "unrelated"));
}

test "encode folds accents and maps unknown words to UNK" {
    var t = try testTokenizer(testing.allocator);
    defer t.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // hèllo folds to hello; an unmatchable word is a single UNK even when a
    // prefix of it matches.
    try testing.expectEqualSlices(u32, &.{1}, try encodeToSlice(&t, a, "hèllo"));
    try testing.expectEqualSlices(u32, &.{0}, try encodeToSlice(&t, a, "zigzag"));
    try testing.expectEqualSlices(u32, &.{0}, try encodeToSlice(&t, a, "🦀"));
}

test "encode handles whitespace variants and empty input" {
    var t = try testTokenizer(testing.allocator);
    defer t.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    try testing.expectEqualSlices(u32, &.{ 1, 2 }, try encodeToSlice(&t, a, "  hello\t\nworld  "));
    try testing.expectEqualSlices(u32, &.{ 1, 2 }, try encodeToSlice(&t, a, "hello\u{00A0}world"));
    try testing.expectEqualSlices(u32, &.{}, try encodeToSlice(&t, a, ""));
    try testing.expectEqualSlices(u32, &.{}, try encodeToSlice(&t, a, "   "));
}

test "initFromJson rejects non-WordPiece models" {
    try testing.expectError(error.UnsupportedTokenizer, Tokenizer.initFromJson(testing.allocator,
        \\{"model":{"type":"BPE","vocab":{}}}
    ));
}
