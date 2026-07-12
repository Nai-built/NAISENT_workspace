using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Windows.Forms;

using Environments;
using Mechanics;
using Extras;

namespace Display_WinForms_;

public class EnvironmentForm : Form
{
    System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();

    HashSet<Keys> activeInputs = new HashSet<Keys>();

    BounceBallEnvironment environment;

    float passedDeltaTime = 0;

    static float DELTA_TIME = .016f*3*2;

    bool pause = false;

    public PreviewForm smallPreview;

    public DownSampler downSampler {get; private set;}

    public EnvironmentForm()
    {
        smallPreview = new PreviewForm();
        smallPreview.Show();

        smallPreview.Text = "NAISENT EYES";

        environment = new BounceBallEnvironment(this);

        Text = "NAISENT ENVIRONMENT";
        ClientSize = new Size(800, 600);
        DoubleBuffered = true;

        timer.Interval = 1; // 16 // ~60 FPS
        timer.Tick += (s, e) =>
        {
            if (pause)
            {
                // environment.startTime = DateTime.Now;
                return;
            }

            bool display = environment.tickEnvironment(DELTA_TIME, activeInputs, passedDeltaTime);
            if (display)
            {
                Invalidate(); // request redraw
            }
            passedDeltaTime += DELTA_TIME;
        };
        timer.Start();
        
        KeyDown += this.OnKeyDown;
        KeyUp += this.OnKeyUp;

        this.KeyPreview = true;

        downSampler = new DownSampler(this, 80, 60);
    }

    void OnKeyDown(object sender, KeyEventArgs e)
    {
        activeInputs.Add(e.KeyCode);

        if (e.KeyCode == Keys.O)
        {
            pause = !pause;
        }
    }
    void OnKeyUp(object sender, KeyEventArgs e)
    {
        activeInputs.Remove(e.KeyCode);
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        Graphics g = e.Graphics;

        // Clear background
        g.Clear(Color.Black);

        environment.displayEnvironment(g, new Vector2(ClientSize.Width, ClientSize.Height));

        this.displayPreview();
        
        // this.environment.isUpdatingState = false;
    }

    public void displayPreview()
    {
        downSampler.update();
        downSampler.FloatArrayToBitmap();
    }

    protected override void OnShown(EventArgs e)
    {
        displayPreview();
    }
}
