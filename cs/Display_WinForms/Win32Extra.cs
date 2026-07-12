using System.Runtime.InteropServices;

namespace Extras;

static class Win32
{
    [DllImport("user32.dll")]
    public static extern bool PrintWindow(
        IntPtr hwnd,
        IntPtr hdcBlt,
        uint nFlags);
}