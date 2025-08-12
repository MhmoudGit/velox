const std = @import("std");

pub const NdArray = struct {
    data: []f32,
    len: usize,

    pub fn init(data: []f32) NdArray {
        return NdArray{
            .data = data,
            .len = data.len,
        };
    }

    pub fn add(self: *NdArray, other: NdArray) !void {
        std.debug.assert(self.len == other.len);

        const VecType = @Vector(4, f32);
        var i: usize = 0;
        const chunk_len = 4;

        // SIMD loop
        while (i + chunk_len <= self.len) : (i += chunk_len) {
            const va: VecType = self.data[i .. i + chunk_len][0..4].*;
            const vb: VecType = other.data[i .. i + chunk_len][0..4].*;

            const sumv = va + vb;
            self.data[i .. i + chunk_len][0..4].* = sumv;
        }

        // Remainder loop
        while (i < self.len) : (i += 1) {
            self.data[i] += other.data[i];
        }
    }

    pub fn sub(self: *NdArray, other: NdArray) !void {
        std.debug.assert(self.len == other.len);

        const VecType = @Vector(4, f32);
        var i: usize = 0;
        const chunk_len = 4;

        // SIMD loop
        while (i + chunk_len <= self.len) : (i += chunk_len) {
            const va: VecType = self.data[i .. i + chunk_len][0..4].*;
            const vb: VecType = other.data[i .. i + chunk_len][0..4].*;

            const subv = va - vb;
            self.data[i .. i + chunk_len][0..4].* = subv;
        }

        // Remainder loop
        while (i < self.len) : (i += 1) {
            self.data[i] -= other.data[i];
        }
    }

    pub fn mul(self: *NdArray, other: NdArray) !void {
        std.debug.assert(self.len == other.len);

        const VecType = @Vector(4, f32);
        var i: usize = 0;
        const chunk_len = 4;

        // SIMD loop
        while (i + chunk_len <= self.len) : (i += chunk_len) {
            const va: VecType = self.data[i .. i + chunk_len][0..4].*;
            const vb: VecType = other.data[i .. i + chunk_len][0..4].*;

            const mulv = va * vb;
            self.data[i .. i + chunk_len][0..4].* = mulv;
        }

        // Remainder loop
        while (i < self.len) : (i += 1) {
            self.data[i] *= other.data[i];
        }
    }

    pub fn div(self: *NdArray, other: NdArray) !void {
        std.debug.assert(self.len == other.len);

        const VecType = @Vector(4, f32);
        var i: usize = 0;
        const chunk_len = 4;

        // SIMD loop
        while (i + chunk_len <= self.len) : (i += chunk_len) {
            const va: VecType = self.data[i .. i + chunk_len][0..4].*;
            const vb: VecType = other.data[i .. i + chunk_len][0..4].*;

            const divv = va / vb;
            self.data[i .. i + chunk_len][0..4].* = divv;
        }

        // Remainder loop
        while (i < self.len) : (i += 1) {
            self.data[i] /= other.data[i];
        }
    }

    pub fn scale(self: *NdArray, scalar: f32, op: enum { add, sub, mul, div }) !void {
        const VecType = @Vector(4, f32);
        const chunk_len = 4;

        switch (op) {
            .add => {
                var i: usize = 0;
                while (i + chunk_len <= self.len) : (i += chunk_len) {
                    const va: VecType = self.data[i .. i + chunk_len][0..4].*;
                    const vb: VecType = @splat(scalar);

                    const scaledv = va + vb;
                    self.data[i .. i + chunk_len][0..4].* = scaledv;
                }

                // Remainder loop
                while (i < self.len) : (i += 1) {
                    self.data[i] += scalar;
                }
            },
            .sub => {
                var i: usize = 0;
                while (i + chunk_len <= self.len) : (i += chunk_len) {
                    const va: VecType = self.data[i .. i + chunk_len][0..4].*;
                    const vb: VecType = @splat(scalar);

                    const scaledv = va - vb;
                    self.data[i .. i + chunk_len][0..4].* = scaledv;
                }

                // Remainder loop
                while (i < self.len) : (i += 1) {
                    self.data[i] -= scalar;
                }
            },
            .mul => {
                var i: usize = 0;
                while (i + chunk_len <= self.len) : (i += chunk_len) {
                    const va: VecType = self.data[i .. i + chunk_len][0..4].*;
                    const vb: VecType = @splat(scalar);

                    const scaledv = va * vb;
                    self.data[i .. i + chunk_len][0..4].* = scaledv;
                }

                // Remainder loop
                while (i < self.len) : (i += 1) {
                    self.data[i] *= scalar;
                }
            },
            .div => {
                var i: usize = 0;
                while (i + chunk_len <= self.len) : (i += chunk_len) {
                    const va: VecType = self.data[i .. i + chunk_len][0..4].*;
                    const vb: VecType = @splat(scalar);

                    const scaledv = va / vb;
                    self.data[i .. i + chunk_len][0..4].* = scaledv;
                }

                // Remainder loop
                while (i < self.len) : (i += 1) {
                    self.data[i] /= scalar;
                }
            },
        }
    }
};
