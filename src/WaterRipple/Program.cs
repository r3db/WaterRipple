using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;

namespace WaterRipple
{
    internal static class Program
    {
        private static void Main()
        {
            const int scale  = 4;
            const int width  = scale * 1920;
            const int height = scale * 960;

            Measure(() => WaterRippleCpu.Render1(width, height), "cpu.1.png", false, "CPU: Native GDI+ Bitmap!");
            Measure(() => WaterRippleCpu.Render2(width, height), "cpu.2.png", false, "CPU: Byte Array!");

            Measure(() => WaterRippleGpu.Render1(width, height), "gpu.1.png", true,  "GPU: Alea Parallel.For!");
            Measure(() => WaterRippleGpu.Render2(width, height), "gpu.2.png", true,  "GPU: Custom!");
            Measure(() => WaterRippleGpu.Render3(width, height), "gpu.3.png", true,  "GPU: Fixed Block Size!");

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<Image> func, string fileName, bool isGpu, string description)
        {
            const string format = "{0,9}";

            Func<Stopwatch, string> formatElapsedTime = w => w.Elapsed.TotalSeconds >= 1
                ? string.Format(CultureInfo.InvariantCulture, format + "  (s)", w.Elapsed.TotalSeconds)
                : w.Elapsed.TotalMilliseconds >= 1
                    ? string.Format(CultureInfo.InvariantCulture, format + " (ms)", w.Elapsed.TotalMilliseconds)
                    : string.Format(CultureInfo.InvariantCulture, format + " (μs)", w.Elapsed.TotalMilliseconds * 1000);

            Action consoleColor = () =>
            {
                Console.ForegroundColor = isGpu
                    ? ConsoleColor.White
                    : ConsoleColor.Cyan;
            };

            var sw1 = Stopwatch.StartNew();
            var result1 = func();
            sw1.Stop();

            // Todo: Bandwith is not relevant for this problem!
            Func<Stopwatch, string> bandwidth = w => string.Format(CultureInfo.InvariantCulture, "{0,8:F4} GB/s", (result1.Width * result1.Height * 3) / (w.Elapsed.TotalMilliseconds * 1000000));

            Console.WriteLine(new string('-', 38));
            Console.WriteLine(description);
            consoleColor();
            Console.WriteLine("{0} - {1} [Cold]", formatElapsedTime(sw1), bandwidth(sw1));
            Console.ResetColor();
            result1.Save(fileName, ImageFormat.Png);

            var sw2 = Stopwatch.StartNew();
            func();
            sw2.Stop();
            consoleColor();
            Console.WriteLine("{0} - {1} [Warm]", formatElapsedTime(sw2), bandwidth(sw2));
            Console.ResetColor();
        }
    }
}