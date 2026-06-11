//! model2vec inference in Zig: static embeddings with no runtime model
//! execution. Embedding a text is a vocabulary lookup and a mean, so it runs
//! in microseconds on a CPU and needs nothing installed.
//!
//! Loads the potion family of models (and anything else model2vec produces
//! with a WordPiece tokenizer) straight from their HuggingFace layout:
//! tokenizer.json + model.safetensors + config.json in one directory.
//!
//!     var model = try m2v.Model.load(gpa, io, "models/potion-base-8M");
//!     defer model.deinit();
//!     const vec = try model.embed(allocator, "some text");
//!
//! Inference matches the reference implementation: tokenize without special
//! tokens, drop [UNK] ids, cap at `max_tokens`, mean-pool the token vectors,
//! L2-normalize when the model config says to.

const std = @import("std");

pub const safetensors = @import("safetensors.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const accents = @import("accents.zig");

pub const Tokenizer = tokenizer.Tokenizer;

pub const LoadError = error{
    ReadFailed,
    BadConfig,
} || safetensors.Error || tokenizer.Error || std.mem.Allocator.Error;

pub const Model = struct {
    gpa: std.mem.Allocator,
    tok: Tokenizer,
    /// rows * dim, row-major; f32 or the model2vec i8 quantized form. I8 is
    /// pooled as raw values: the global quantization scale cancels under L2
    /// normalization, matching the reference implementation.
    embeddings: safetensors.Matrix,
    rows: usize,
    dim: usize,
    normalize: bool,
    /// Token cap per text, matching the reference default.
    max_tokens: usize = 512,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir_path: []const u8) LoadError!Model {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        const tok_bytes = readFile(a, io, dir_path, "tokenizer.json") catch return error.ReadFailed;
        const st_bytes = readFile(a, io, dir_path, "model.safetensors") catch return error.ReadFailed;

        // `normalize` defaults to true; missing config.json is fine.
        const norm = blk: {
            const cfg_bytes = readFile(a, io, dir_path, "config.json") catch break :blk true;
            const cfg = std.json.parseFromSliceLeaky(std.json.Value, a, cfg_bytes, .{}) catch
                return error.BadConfig;
            if (cfg != .object) return error.BadConfig;
            const v = cfg.object.get("normalize") orelse break :blk true;
            break :blk if (v == .bool) v.bool else true;
        };

        return loadFromBytes(gpa, tok_bytes, st_bytes, .{ .normalize = norm });
    }

    /// Load from in-memory file contents, for models shipped inside the
    /// binary via @embedFile. `normalize` mirrors config.json's `normalize`
    /// key; the potion family uses true.
    pub fn loadFromBytes(
        gpa: std.mem.Allocator,
        tokenizer_json: []const u8,
        safetensors_bytes: []const u8,
        options: struct { normalize: bool = true },
    ) LoadError!Model {
        var tok = try Tokenizer.initFromJson(gpa, tokenizer_json);
        errdefer tok.deinit();

        const emb = try safetensors.parse(gpa, safetensors_bytes);

        if (tok.unk_id >= emb.rows) return error.BadShape;

        return .{
            .gpa = gpa,
            .tok = tok,
            .embeddings = emb.matrix,
            .rows = emb.rows,
            .dim = emb.cols,
            .normalize = options.normalize,
        };
    }

    pub fn deinit(self: *Model) void {
        self.tok.deinit();
        self.embeddings.deinit(self.gpa);
        self.* = undefined;
    }

    /// Embed `text` into a freshly allocated vector of `dim` values.
    /// The model is read-only here, so concurrent embeds are fine as long as
    /// each call gets its own allocator.
    pub fn embed(self: *const Model, allocator: std.mem.Allocator, text: []const u8) ![]f32 {
        const out = try allocator.alloc(f32, self.dim);
        errdefer allocator.free(out);
        try self.embedInto(allocator, text, out);
        return out;
    }

    /// Embed into a caller-owned buffer of exactly `dim` values. `scratch` is
    /// for tokenization temporaries and is fully released before returning,
    /// so any allocator works; an arena is merely the fastest choice.
    pub fn embedInto(self: *const Model, scratch: std.mem.Allocator, text: []const u8, out: []f32) !void {
        std.debug.assert(out.len == self.dim);

        // The reference bounds the input by characters before tokenizing so
        // pathological inputs don't pay full tokenization cost.
        const bounded = truncateChars(text, self.max_tokens * self.tok.median_token_len);

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(scratch);
        try self.tok.encode(scratch, bounded, &ids);

        @memset(out, 0);
        const used = switch (self.embeddings) {
            .f32_data => |m| self.pool(f32, m, ids.items, out),
            .i8_data => |m| self.pool(i8, m, ids.items, out),
        };
        if (used == 0) return; // no known tokens: zero vector, like the reference

        const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(used));
        for (out) |*o| o.* *= inv_n;

        if (self.normalize) {
            var sum_sq: f32 = 0;
            for (out) |o| sum_sq += o * o;
            if (sum_sq > 0) {
                const inv_norm = 1.0 / @sqrt(sum_sq);
                for (out) |*o| o.* *= inv_norm;
            }
        }
    }

    fn pool(self: *const Model, comptime T: type, matrix: []const T, ids: []const u32, out: []f32) usize {
        var used: usize = 0;
        for (ids) |id| {
            if (id == self.tok.unk_id) continue;
            if (used == self.max_tokens) break;
            if (id >= self.rows) continue;
            const row = matrix[@as(usize, id) * self.dim ..][0..self.dim];
            for (out, row) |*o, r| {
                o.* += if (T == f32) r else @floatFromInt(r);
            }
            used += 1;
        }
        return used;
    }
};

fn readFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(256 * 1024 * 1024));
}

/// Slice `s` to at most `max` codepoints without splitting a sequence.
fn truncateChars(s: []const u8, max: usize) []const u8 {
    var count: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        if (count == max) return s[0..i];
        const len = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
        i = @min(i + len, s.len);
        count += 1;
    }
    return s;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test {
    testing.refAllDecls(@This());
    _ = @import("quantize.zig");
}

test "truncateChars respects codepoint boundaries" {
    try testing.expectEqualStrings("hél", truncateChars("héllo", 3));
    try testing.expectEqualStrings("héllo", truncateChars("héllo", 99));
    try testing.expectEqualStrings("", truncateChars("héllo", 0));
}

// Parity against the reference implementation. Runs only when a real model
// is present (scripts/fetch-model.sh puts it there); unit tests above cover
// the pieces without it.
test "matches reference embeddings for potion-base-8M" {
    try parityCheck("models/potion-base-8M", "testdata/golden.json");
}

test "matches reference embeddings for the i8-quantized model" {
    try parityCheck("models/potion-base-8M-i8", "testdata/golden_i8.json");
}

fn parityCheck(comptime model_dir: []const u8, comptime golden_path: []const u8) !void {
    std.Io.Dir.cwd().access(testing.io, model_dir ++ "/model.safetensors", .{}) catch return error.SkipZigTest;

    var model = try Model.load(testing.allocator, testing.io, model_dir);
    defer model.deinit();

    try testing.expectEqual(@as(usize, 256), model.dim);
    try testing.expect(model.normalize);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const golden_bytes = try std.Io.Dir.cwd().readFileAlloc(testing.io, golden_path, a, .limited(16 * 1024 * 1024));
    const golden = try std.json.parseFromSliceLeaky(std.json.Value, a, golden_bytes, .{});

    for (golden.array.items) |case| {
        const text = case.object.get("text").?.string;
        const want = case.object.get("vec").?.array.items;

        const got = try model.embed(a, text);
        try testing.expectEqual(want.len, got.len);

        var max_diff: f64 = 0;
        for (got, want) |g, w| {
            const wf = switch (w) {
                .float => |f| f,
                .integer => |i| @as(f64, @floatFromInt(i)),
                else => unreachable,
            };
            max_diff = @max(max_diff, @abs(@as(f64, g) - wf));
        }
        if (max_diff > 1e-5) {
            std.debug.print("parity failure ({s}) on \"{s}\": max diff {d}\n", .{ model_dir, text, max_diff });
            return error.ParityMismatch;
        }
    }
}
