const std = @import("std");
const ndarray = @import("ndarray.zig").NdArray;

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

test "add" {
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 10, 20, 30, 40, 50, 60 };

    var x = ndarray.init(&a_data);
    const y = ndarray.init(&b_data);
    try x.add(y);

    try std.testing.expectEqual(x.data[0], 11.0);
    try std.testing.expectEqual(x.data[1], 22.0);
    try std.testing.expectEqual(x.data[2], 33.0);
    try std.testing.expectEqual(x.data[3], 44.0);
}

test "sub" {
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 10, 20, 30, 40, 50, 60 };

    var x = ndarray.init(&a_data);
    const y = ndarray.init(&b_data);
    try x.sub(y);

    try std.testing.expectEqual(x.data[0], -9.0);
    try std.testing.expectEqual(x.data[1], -18.0);
    try std.testing.expectEqual(x.data[2], -27.0);
    try std.testing.expectEqual(x.data[3], -36.0);
}

test "mul" {
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 10, 20, 30, 40, 50, 60 };

    var x = ndarray.init(&a_data);
    const y = ndarray.init(&b_data);
    try x.mul(y);

    try std.testing.expectEqual(x.data[0], 10.0);
    try std.testing.expectEqual(x.data[1], 40.0);
    try std.testing.expectEqual(x.data[2], 90.0);
    try std.testing.expectEqual(x.data[3], 160.0);
}

test "div" {
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 10, 20, 30, 40, 50, 60 };

    var x = ndarray.init(&a_data);
    const y = ndarray.init(&b_data);
    try x.div(y);

    try std.testing.expectEqual(x.data[0], 0.1);
    try std.testing.expectEqual(x.data[1], 0.1);
    try std.testing.expectEqual(x.data[2], 0.1);
    try std.testing.expectEqual(x.data[3], 0.1);
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

test "vector_add_scalar" {
    var x = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var y = ndarray.init(&x);
    try y.scale(1.0, .add);

    try std.testing.expectEqual(y.data[0], 2);
    try std.testing.expectEqual(y.data[1], 3);
    try std.testing.expectEqual(y.data[2], 4);
    try std.testing.expectEqual(y.data[3], 5);
}

test "vector_sub_scalar" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y = ndarray.init(&x);
    try y.scale(1.0, .sub);

    try std.testing.expectEqual(y.data[0], 0.0);
    try std.testing.expectEqual(y.data[1], 1.0);
    try std.testing.expectEqual(y.data[2], 2.0);
    try std.testing.expectEqual(y.data[3], 3.0);
}

test "vector_mul_scalar" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y = ndarray.init(&x);
    try y.scale(2.0, .mul);

    try std.testing.expectEqual(y.data[0], 2.0);
    try std.testing.expectEqual(y.data[1], 4.0);
    try std.testing.expectEqual(y.data[2], 6.0);
    try std.testing.expectEqual(y.data[3], 8.0);
}

test "vector_div_scalar" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y = ndarray.init(&x);
    try y.scale(2.0, .div);

    try std.testing.expectEqual(y.data[0], 0.5);
    try std.testing.expectEqual(y.data[1], 1.0);
    try std.testing.expectEqual(y.data[2], 1.5);
    try std.testing.expectEqual(y.data[3], 2.0);
}

pub fn deriv(
    allocator: std.mem.Allocator,
    len: comptime_int,
    input: []f32,
    delta: ?f32,
    callback: fn ([]f32, comptime_int) [len]f32,
) ![]f32 {
    const delta_def = if (delta == null or delta == 0) 0.001 else delta.?;

    // Create two modified input copies
    var plus: [len]f32 = undefined;
    var minus: [len]f32 = undefined;
    @memcpy(&plus, input);
    @memcpy(&minus, input);

    var x = ndarray.init(&plus);
    var y = ndarray.init(&minus);

    try x.scale(delta_def, .add);
    try y.scale(delta_def, .sub);

    var a = callback(x.data, len);
    var b = callback(y.data, len);

    var right = ndarray.init(&a);
    const left = ndarray.init(&b);

    try right.sub(left);
    try right.scale(2 * delta_def, .div);

    const result = try allocator.create([len]f32);
    @memcpy(result, right.data);

    return result;
}

test "deriv" {
    const allocator = std.heap.page_allocator;
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = try deriv(allocator, x.len, &x, null, square_vec);
    defer allocator.free(y);

    try std.testing.expectApproxEqAbs(2.0, y[0], 0.001);
    try std.testing.expectApproxEqAbs(4.0, y[1], 0.001);
    try std.testing.expectApproxEqAbs(6.0, y[2], 0.001);
    try std.testing.expectApproxEqAbs(8.0, y[3], 0.001);
}

// pub fn chains(input: []f32, len: comptime_int) [len]f32 {
//     const funcs = [2]fn ([]f32, comptime_int) [len]f32{
//         square_vec,
//         square_vec,
//     };

//     var output: [len]f32 = undefined;
//     @memcpy(&output, input);

//     const f1 = funcs[0];
//     const f2 = funcs[1];

//     var v1 = f2(&output, len);
//     const v2 = f1(&v1, len);
//     return v2;
// }

// test "chains" {
//     var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
//     const y = chains(&x, x.len);
//     try std.testing.expectApproxEqAbs(y[0], 1.0, 0.001);
//     try std.testing.expectApproxEqAbs(y[1], 16.0, 0.001);
//     try std.testing.expectApproxEqAbs(y[2], 81.0, 0.001);
//     try std.testing.expectApproxEqAbs(y[3], 256.0, 0.001);
// }
