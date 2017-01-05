using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

namespace WaterRipple
{
    internal static class Program
    {
        private static void Main()
        {
            const int scale  = 8;
            const int width  = scale * 1920;
            const int height = scale * 960;

            Measure(() => WaterRipple.RenderCpu1(width, height), "ripple.cpu.1.png", "CPU: Using Native GDI+ Bitmap!");
            Measure(() => WaterRipple.RenderCpu2(width, height), "ripple.cpu.2.png", "CPU: Using byte Array!");
            Measure(() => WaterRipple.RenderGpu1(width, height), "ripple.gpu.1.png", "GPU: Using byte Array!");
            Measure(() => WaterRipple.RenderGpu2(width, height), "ripple.gpu.2.png", "GPU: Allocating Memory on GPU only!");
            Measure(() => WaterRipple.RenderGpu3(width, height), "ripple.gpu.3.png", "GPU: Parallel.For!");

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<Image> func, string fileName, string description)
        {
            Func<Stopwatch, string> formatElapsedTime = (watch) => watch.Elapsed.TotalSeconds >= 1
                ? $"{watch.Elapsed.TotalSeconds}s"
                : $"{watch.ElapsedMilliseconds}ms";

            var sw1 = Stopwatch.StartNew();
            var bmp1 = func();
            sw1.Stop();

            Console.WriteLine(new string('-', 43));
            Console.WriteLine(description);
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("{0} [Cold]", formatElapsedTime(sw1));
            Console.ResetColor();
            bmp1.Save(fileName, ImageFormat.Png);

            Console.WriteLine();

            var sw2 = Stopwatch.StartNew();
            func();
            sw2.Stop();
            Console.WriteLine(description);
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("{0} [Warm]", formatElapsedTime(sw2));
            Console.ResetColor();

            Console.WriteLine(new string('-', 43));
            Console.WriteLine();
        }
    }
}