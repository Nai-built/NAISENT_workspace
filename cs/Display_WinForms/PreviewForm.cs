public class PreviewForm : Form
{
    private Bitmap _bmp;

    public PreviewForm()
    {
        DoubleBuffered = true;
        ClientSize = new Size(400, 300); // any size you want
    }

    public void SetBitmap(Bitmap bmp)
    {
        _bmp = bmp;
        Invalidate();
    }

    protected override void OnPaintBackground(PaintEventArgs e)
    {
        // prevent default background erase
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        if (_bmp == null)
            return;

        e.Graphics.Clear(Color.Black);

        e.Graphics.InterpolationMode =
            System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;

        e.Graphics.PixelOffsetMode =
            System.Drawing.Drawing2D.PixelOffsetMode.None;

        var re = ClientRectangle;

        e.Graphics.DrawImage(
            _bmp,
            re,
            new Rectangle(0, 0, _bmp.Width, _bmp.Height),
            GraphicsUnit.Pixel);
    }
}
