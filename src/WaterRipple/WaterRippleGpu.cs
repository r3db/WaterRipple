using System;
using System.Drawing;
using Alea;
using Alea.CSharp;
using Alea.Parallel;

namespace WaterRipple
{
    internal static class WaterRippleGpu
    {
        private const int ColorComponents = 3;

        // Alea Parallel.For!
        internal static Image Render1(int width, int height)
        {
            var resultLength = ColorComponents * width * height;
            var resultMemory = Gpu.Default.AllocateDevice<byte>(resultLength);
            var resultDevPtr = new deviceptr<byte>(resultMemory.Handle);

            Gpu.Default.For(0, width * height, i =>
            {
                ComputeRippleAtOffset(resultDevPtr, i, width, height);
            });

            return BitmapUtility.FromByteArray(Gpu.CopyToHost(resultMemory), width, height);
        }

        // Custom!
        internal static Image Render2(int width, int height)
        {
            var resultLength = ColorComponents * width * height;
            var resultMemory = Gpu.Default.AllocateDevice<byte>(resultLength);
            var resultDevPtr = new deviceptr<byte>(resultMemory.Handle);

            var lp = ComputeLaunchParameters(width, height);

            Gpu.Default.Launch(() =>
            {
                var i = blockDim.x * blockIdx.x + threadIdx.x;
                ComputeRippleAtOffset(resultDevPtr, i, width, height);
            }, lp);

            return BitmapUtility.FromByteArray(Gpu.CopyToHost(resultMemory), width, height);
        }

        // Fixed Block Size!
        internal static Image Render3(int width, int height)
        {
            var resultLength = ColorComponents * width * height;
            var resultMemory = Gpu.Default.AllocateDevice<byte>(resultLength);
            var resultDevPtr = new deviceptr<byte>(resultMemory.Handle);

            var lp = new LaunchParam(256, 256);

            Gpu.Default.Launch(() =>
            {
                var i = blockDim.x * blockIdx.x + threadIdx.x;

                while (ColorComponents * i < resultLength)
                {
                    ComputeRippleAtOffset(resultDevPtr, i, width, height);
                    i += gridDim.x * blockDim.x;
                }
            }, lp);

            return BitmapUtility.FromByteArray(Gpu.CopyToHost(resultMemory), width, height);
        }

        private static void ComputeRippleAtOffset(deviceptr<byte> result, int index, int width, int height)
        {
            var x = index % width;
            var y = index / width;
            var offset = ColorComponents * index;

            if (offset < 3 * width * height)
            {
                var fx = x - width  * 0.5f;
                var fy = y - height * 0.5f;

                var d = DeviceFunction.Sqrt(fx * fx + fy * fy);
                var v = (byte)(128f + 127f * DeviceFunction.Cos(d / 10f) / (d / 10f + 1f));

                result[offset + 0] = v;
                result[offset + 1] = v;
                result[offset + 2] = v;
            }
        }

        private static LaunchParam ComputeLaunchParameters(int width, int height)
        {
            const int tpb = 256;
            return new LaunchParam((ColorComponents * width * height + (tpb - 1)) / tpb, tpb);
        }
    }
}