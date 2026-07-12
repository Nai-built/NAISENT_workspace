using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Windows.Forms;

namespace Extras;

public class DownSampler
{
    private object bitsLockedMutex = new object();

    int width;
    int height;
    int channelSize;
    int C = 3; // R, G, B

    // Create a new Bitmap
    public Bitmap tensorView {get; private set;}
        
    public void FloatArrayToBitmap()
    {
        // Create a byte array for pixel data (4 bytes per pixel for ARGB)
        int bytesPerPixel = 4;
        byte[] pixelBytes = new byte[channelSize * bytesPerPixel];

        // Determine the min/max values in the float array for scaling (optional, if values aren't 0.0-1.0)
        // float minVal = floatArray.Min();
        // float maxVal = floatArray.Max();
        // float range = maxVal - minVal;

        for (int i = 0; i < channelSize; i++)
        {
            byte b = (byte)(tensor[i + channelSize*2]*255.0f);
            byte g = (byte)(tensor[i + channelSize]*255.0f);
            byte r = (byte)(tensor[i]*255.0f);
            
            int byteIndex = i * bytesPerPixel;

            // Set ARGB values: A=255 (fully opaque), R=grayValue, G=grayValue, B=grayValue
            pixelBytes[byteIndex + 0] = b; // Blue component (BMP stores as BGRA)
            pixelBytes[byteIndex + 1] = g; // Green component
            pixelBytes[byteIndex + 2] = r; // Red component
            pixelBytes[byteIndex + 3] = 255;       // Alpha component
        }

        // Lock the bitmap data to write pixel bytes directly
        BitmapData bmpData = tensorView.LockBits(new Rectangle(0, 0, width, height),
                                            ImageLockMode.WriteOnly,
                                            tensorView.PixelFormat);
        
        // Copy the byte array to the bitmap's Scan0 pointer
        System.Runtime.InteropServices.Marshal.Copy(pixelBytes, 0, bmpData.Scan0, pixelBytes.Length);

        // Unlock the bits
        tensorView.UnlockBits(bmpData);
    }
    /// <summary>
    /// Converts a bitmap to a [3,H,W] tensor (channel-first) normalized to [0,1] ( BY ChatGPT )
    /// </summary>
    public float[,,] BitmapToRGBTensor_CHW()
    {
        float[,,] tensor;
        lock (bitsLockedMutex)
        {
            int w = targetMap.Width;
            int h = targetMap.Height;
            tensor = new float[3, h, w];

            var rect = new Rectangle(0, 0, w, h);
            var data = targetMap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            int bytes = Math.Abs(data.Stride) * h;
            byte[] buffer = new byte[bytes];
            System.Runtime.InteropServices.Marshal.Copy(data.Scan0, buffer, 0, bytes);

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int i = y * data.Stride + x * 4;

                    byte b = buffer[i + 0];
                    byte g = buffer[i + 1];
                    byte r = buffer[i + 2];

                    tensor[0, y, x] = r / 255f;
                    tensor[1, y, x] = g / 255f;
                    tensor[2, y, x] = b / 255f;
                }
            }

            targetMap.UnlockBits(data);
        }
        return tensor;
    }

    public void BitmapToRGBTensor_CHW_Jagged()
    {
        lock (bitsLockedMutex)
        {
            // lock bitmap for reading
            var rect = new Rectangle(0, 0, width, height);

            var data = targetMap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            int bytes = Math.Abs(data.Stride) * height;
            byte[] buffer = new byte[bytes];
            System.Runtime.InteropServices.Marshal.Copy(data.Scan0, buffer, 0, bytes);

            // fill jagged array
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int i = y * data.Stride + x * 4;

                    byte b = buffer[i + 0];
                    byte g = buffer[i + 1];
                    byte r = buffer[i + 2];

                    // tensor[(y*width + x)*C + 0] = r / 255f; // R channel
                    // tensor[(y*width + x)*C + 1] = g / 255f; // G channel
                    // tensor[(y*width + x)*C + 2] = b / 255f; // B channel

                    tensor[x + y*width] = r / 255.0f;
                    tensor[x + y*width + this.channelSize] = g / 255.0f;
                    tensor[x + y*width + this.channelSize*2] = b / 255.0f;
                }
            }

            targetMap.UnlockBits(data);
        }
    }

    private Bitmap full;
    private Bitmap client;
    private Rectangle crop;
    public Bitmap targetMap;

    private Form form;

    public float[] tensor;

    public DownSampler(Form _form, int downSampleX, int downSampleY)
    {
        form = _form;

        var clientRect = form.ClientRectangle;
        var clientScreen = form.RectangleToScreen(clientRect);
        var windowScreen = new Rectangle(form.Left, form.Top, form.Width, form.Height);

        full = new Bitmap(form.Width, form.Height, PixelFormat.Format32bppArgb);

        crop = new Rectangle(
            clientScreen.Left - windowScreen.Left,
            clientScreen.Top - windowScreen.Top,
            clientRect.Width,
            clientRect.Height);

        targetMap = new Bitmap(80, 60, PixelFormat.Format32bppArgb);

        this.width = targetMap.Width;
        this.height = targetMap.Height;
        this.channelSize = this.width*this.height;

        this.tensor = new float[C*width*height];

        tensorView = new Bitmap(width, height, PixelFormat.Format32bppArgb);

        client = new Bitmap(form.ClientSize.Width, form.ClientSize.Height, PixelFormat.Format32bppArgb);
    }

    public void update()
    {
        using (var g = Graphics.FromImage(full))
        {
            IntPtr hdc = g.GetHdc();
            Win32.PrintWindow(form.Handle, hdc, 0);
            g.ReleaseHdc(hdc);
        }

        using (var g = Graphics.FromImage(client))
        {
            g.DrawImage(
                full,
                new Rectangle(0, 0, client.Width, client.Height),  // dest
                crop,                                                          // source
                GraphicsUnit.Pixel
            );
        }

        lock (bitsLockedMutex)
        {
            using (var g = Graphics.FromImage(targetMap))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBilinear;
                g.DrawImage(client, 0, 0, 80, 60);
            }
        }

        BitmapToRGBTensor_CHW_Jagged();
    }

    public void desktopUpdate()
    {
        var clientTopLeft = form.PointToScreen(Point.Empty);
        var size = form.ClientSize;

        using (var g = Graphics.FromImage(client))
        {
            g.CopyFromScreen(
                clientTopLeft,
                Point.Empty,
                size,
                CopyPixelOperation.SourceCopy);
        }

        lock (bitsLockedMutex)
        {
            using (var g = Graphics.FromImage(targetMap))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBilinear;
                g.DrawImage(client, 0, 0, 80, 60);
            }
        }
    }
}