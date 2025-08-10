const std = @import("std");

pub fn square(x: i32) i32 {
    return x * x;
}

pub fn relu(x: i32) i32 {
    if (x < 0) {
        return 0;
    }
    return x;
}

pub fn square_vec(data: []f32, len: comptime_int) [len]f32 {
    var x: [len]f32 = undefined;
    @memcpy(&x, data);
    const vec_x: @Vector(len, f32) = x;
    return vec_x * vec_x;
}

test "square_vec" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = square_vec(&x, x.len);

    try std.testing.expectEqual(y[0], 1.0);
    try std.testing.expectEqual(y[1], 4.0);
    try std.testing.expectEqual(y[2], 9.0);
    try std.testing.expectEqual(y[3], 16.0);
}

pub fn add_vec(data_x: []const f32, data_y: []const f32, len: comptime_int) [len]f32 {
    var x: [len]f32 = undefined;
    @memcpy(&x, data_x);
    var y: [len]f32 = undefined;
    @memcpy(&y, data_y);
    const vec_x: @Vector(len, f32) = x;
    const vec_y: @Vector(len, f32) = y;
    return vec_x + vec_y;
}

test "add_vec" {
    const len = 4;
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    const z = add_vec(&x, &y, len);

    try std.testing.expectEqual(z[0], 6.0);
    try std.testing.expectEqual(z[1], 8.0);
    try std.testing.expectEqual(z[2], 10.0);
    try std.testing.expectEqual(z[3], 12.0);
}

pub fn sub_vec(data_x: []const f32, data_y: []const f32, len: comptime_int) [len]f32 {
    var x: [len]f32 = undefined;
    @memcpy(&x, data_x);
    var y: [len]f32 = undefined;
    @memcpy(&y, data_y);
    const vec_x: @Vector(len, f32) = x;
    const vec_y: @Vector(len, f32) = y;
    return vec_x - vec_y;
}

test "sub_vec" {
    const len = 4;
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    const z = sub_vec(&x, &y, len);

    try std.testing.expectEqual(z[0], -4.0);
    try std.testing.expectEqual(z[1], -4.0);
    try std.testing.expectEqual(z[2], -4.0);
    try std.testing.expectEqual(z[3], -4.0);
}

pub fn mul_vec(data_x: []const f32, data_y: []const f32, len: comptime_int) [len]f32 {
    var x: [len]f32 = undefined;
    @memcpy(&x, data_x);
    var y: [len]f32 = undefined;
    @memcpy(&y, data_y);
    const vec_x: @Vector(len, f32) = x;
    const vec_y: @Vector(len, f32) = y;
    return vec_x * vec_y;
}

test "mul_vec" {
    const len = 4;
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    const z = mul_vec(&x, &y, len);

    try std.testing.expectEqual(z[0], 5.0);
    try std.testing.expectEqual(z[1], 12.0);
    try std.testing.expectEqual(z[2], 21.0);
    try std.testing.expectEqual(z[3], 32.0);
}

pub fn div_vec(data_x: []const f32, data_y: []const f32, len: comptime_int) [len]f32 {
    var x: [len]f32 = undefined;
    @memcpy(&x, data_x);
    var y: [len]f32 = undefined;
    @memcpy(&y, data_y);
    const vec_x: @Vector(len, f32) = x;
    const vec_y: @Vector(len, f32) = y;
    return vec_x / vec_y;
}

test "div_vec" {
    const len = 4;
    const x = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    const y = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const z = div_vec(&x, &y, len);

    try std.testing.expectEqual(z[0], 2.0);
    try std.testing.expectEqual(z[1], 2.0);
    try std.testing.expectEqual(z[2], 2.0);
    try std.testing.expectEqual(z[3], 2.0);
}

pub fn leaky_relu_vec(x: []f32) []f32 {
    for (x, 0..) |v, i| {
        if (v < 0) {
            x[i] = v * 0.2;
        }
    }
    return x;
}

test "leaky_relu_vec" {
    var x = [_]f32{ -1.0, 2.0, -3.0, 4.0 };
    const y = leaky_relu_vec(&x);

    try std.testing.expectEqual(y[0], -0.2);
    try std.testing.expectEqual(y[1], 2.0);
    try std.testing.expectEqual(y[2], -0.6);
    try std.testing.expectEqual(y[3], 4.0);
}

pub fn vector_with_scalar(x: []f32, scalar: f32, op: enum { add, sub, mul, div }) []f32 {
    for (x, 0..) |v, i| {
        switch (op) {
            .add => x[i] = v + scalar,
            .sub => x[i] = v - scalar,
            .mul => x[i] = v * scalar,
            .div => x[i] = v / scalar,
        }
    }
    return x;
}

test "vector_add_scalar" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = vector_with_scalar(&x, 1.0, .add);

    try std.testing.expectEqual(y[0], 2.0);
    try std.testing.expectEqual(y[1], 3.0);
    try std.testing.expectEqual(y[2], 4.0);
    try std.testing.expectEqual(y[3], 5.0);
}

test "vector_sub_scalar" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = vector_with_scalar(&x, 1.0, .sub);

    try std.testing.expectEqual(y[0], 0.0);
    try std.testing.expectEqual(y[1], 1.0);
    try std.testing.expectEqual(y[2], 2.0);
    try std.testing.expectEqual(y[3], 3.0);
}

test "vector_mul_scalar" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = vector_with_scalar(&x, 2.0, .mul);

    try std.testing.expectEqual(y[0], 2.0);
    try std.testing.expectEqual(y[1], 4.0);
    try std.testing.expectEqual(y[2], 6.0);
    try std.testing.expectEqual(y[3], 8.0);
}

test "vector_div_scalar" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = vector_with_scalar(&x, 2.0, .div);

    try std.testing.expectEqual(y[0], 0.5);
    try std.testing.expectEqual(y[1], 1.0);
    try std.testing.expectEqual(y[2], 1.5);
    try std.testing.expectEqual(y[3], 2.0);
}

pub fn deriv(len: comptime_int, input: []f32, delta: ?f32, callback: fn ([]f32, comptime_int) [len]f32) []f32 {
    const delta_def = if (delta == null or delta == 0) 0.001 else delta.?;

    // Create two modified input copies
    var plus: [len]f32 = undefined;
    var minus: [len]f32 = undefined;
    @memcpy(&plus, input);
    @memcpy(&minus, input);

    _ = vector_with_scalar(&plus, delta_def, .add);
    _ = vector_with_scalar(&minus, delta_def, .sub);

    const plus_callback = callback(&plus, len);
    const minus_callback = callback(&minus, len);

    var nominator = sub_vec(&plus_callback, &minus_callback, len);
    const denominator = 2 * delta_def;

    const nominator_div = vector_with_scalar(&nominator, denominator, .div);

    return nominator_div;
}

test "deriv" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = deriv(x.len, &x, null, square_vec);

    try std.testing.expectApproxEqAbs(y[0], 2.0, 0.001);
    try std.testing.expectApproxEqAbs(y[1], 4.0, 0.001);
    try std.testing.expectApproxEqAbs(y[2], 6.0, 0.001);
    try std.testing.expectApproxEqAbs(y[3], 8.0, 0.001);
}
