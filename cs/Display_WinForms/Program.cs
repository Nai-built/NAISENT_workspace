namespace Display_WinForms_;
using System.Runtime.InteropServices; // Required for DllImport

static class Program
{
    /// <summary>
    ///  The main entry point for the application.
    /// </summary>
    /// 
    // This imports the native Windows system command to link back to the terminal
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AttachConsole(int dwProcessId);
    private const int ATTACH_PARENT_PROCESS = -1;
    /// 
    [STAThread]
    static void Main(string[] args)
    {
        // Force the app to attach to the terminal it was launched from
        AttachConsole(ATTACH_PARENT_PROCESS);

        // To customize application configuration such as set high DPI settings or default font,
        // see https://aka.ms/applicationconfiguration.
        ApplicationConfiguration.Initialize();

        if (args.Length > 0)
        {
            string _environmentType = args[0].ToLower();
            Application.Run(new EnvironmentForm(_environmentType));
        }
    }    
}