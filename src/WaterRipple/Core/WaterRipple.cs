using System;
using System.Drawing;
using System.Threading.Tasks;
using Alea;
using Alea.CSharp;
using Alea.Parallel;

namespace WaterRipple
{
    internal static class WaterRipple
    {
        // CPU: Using Native GDI+ Bitmap!
        internal static Image RenderCpu1(int width, int height)
        {
            var result = new FastBitmap(width, height);

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    var fx = x - width  * 0.5f;
                    var fy = y - height * 0.5f;
                    var d = (float)Math.Sqrt(fx * fx + fy * fy);
                    var g = (byte)(128f + 127f * Math.Cos(d / 10f) / (d / 10f + 1f));

                    result.SetPixel(x, y, Color.FromArgb(g, g, g));
                }
            });

            return result.Bitmap;
        }

        // CPU: Using byte Array!
        internal static Image RenderCpu2(int width, int height)
        {
            var result = new byte[3 * width * height];
           
            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    ComputeRippleAtOffset(result, x, y, width, height);
                }
            });

            return FastBitmap.FromByteArray(result, width, height);
        }

        // GPU: Using byte Array!
        internal static Image RenderGpu1(int width, int height)
        {
            var result = new byte[3 * width * height];
            var lp = ComputeLaunchParameters(width, height);

            Gpu.Default.Launch(() =>
            {
                var x = blockDim.x * blockIdx.x + threadIdx.x;
                var y = blockDim.y * blockIdx.y + threadIdx.y;

                ComputeRippleAtOffset(result, x, y, width, height);
            }, lp);

            return FastBitmap.FromByteArray(result, width, height);
        }

        // GPU: Allocating Memory on GPU only!
        internal static Image RenderGpu2(int width, int height)
        {
            var deviceResult = Gpu.Default.Allocate<byte>(3 * width * height);
            var lp = ComputeLaunchParameters(width, height);

            Gpu.Default.Launch(() =>
            {
                var x = blockDim.x * blockIdx.x + threadIdx.x;
                var y = blockDim.y * blockIdx.y + threadIdx.y;

                ComputeRippleAtOffset(deviceResult, x, y, width, height);
            }, lp);

            return FastBitmap.FromByteArray(Gpu.CopyToHost(deviceResult), width, height);
        }

        // GPU: Parallel.For!
        internal static Image RenderGpu3(int width, int height)
        {
            var result = new byte[3 * width * height];

            Gpu.Default.For(0, width * height, i =>
            {
                var x = i % width;
                var y = i / width;
                var offset = 3 * i;

                ComputeRippleAtOffset(result, x, y, width, height);
            });

            return FastBitmap.FromByteArray(result, width, height);
        }

        // ReSharper disable once SuggestBaseTypeForParameter
        private static void ComputeRippleAtOffset(byte[] result, int x, int y, int width, int height)
        {
            var fx = x - width  * 0.5f;
            var fy = y - height * 0.5f;

            var d = DeviceFunction.Sqrt(fx * fx + fy * fy);
            var g = (byte)(128f + 127f * DeviceFunction.Cos(d / 10f) / (d / 10f + 1f));

            var offset = 3 * (y * width + x);

            result[offset + 0] = g;
            result[offset + 1] = g;
            result[offset + 2] = g;
        }

        private static LaunchParam ComputeLaunchParameters(int width, int height)
        {
            const int threads = 32;
            return new LaunchParam(new dim3((width + (threads - 1)) / threads, (height + (threads - 1)) / threads), new dim3(threads, threads));
        }
    }
}