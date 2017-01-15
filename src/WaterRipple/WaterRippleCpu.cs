using System;
using System.Drawing;
using System.Threading.Tasks;
using Alea;

namespace WaterRipple
{
    internal static class WaterRippleCpu
    {
        private const int ColorComponents = 3;

        // Native GDI+ Bitmap!
        internal static Image Render1(int width, int height)
        {
            var result = new FastBitmap(width, height);

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    ComputeRippleAtOffset(x, y, width, height, v => result.SetPixel(x, y, Color.FromArgb(v, v, v)));
                }
            });

            return result.Bitmap;
        }

        // Byte Array!
        internal static Image Render2(int width, int height)
        {
            var result = new byte[ColorComponents * width * height];

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    ComputeRippleAtOffset(x, y, width, height, v =>
                    {
                        var offset = ColorComponents * (y * width + x);

                        result[offset + 0] = v;
                        result[offset + 1] = v;
                        result[offset + 2] = v;
                    });
                }
            });

            return BitmapUtility.FromByteArray(result, width, height);
        }

        // Helpers!
        private static void ComputeRippleAtOffset(int x, int y, int width, int height, Action<byte> action)
        {
            var fx = x - width  * 0.5f;
            var fy = y - height * 0.5f;

            var d = DeviceFunction.Sqrt(fx * fx + fy * fy);
            var v = (byte)(128f + 127f * DeviceFunction.Cos(d / 10f) / (d / 10f + 1f));

            action(v);
        }
    }
}