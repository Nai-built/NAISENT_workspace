using Mechanics;
using Display_WinForms_;

namespace Environments
{
    public partial class ShapeTestEnvironment
    {
        public static string[] SHAPES = new string[] {"Square", "Circle", "Triangle"};

        public EnvironmentForm _Form {get; private set;}
        private GameObject shapeObject;

        public string targetShape {get; private set;}

        public EnvironmentTick tickEnvironment {get; private set;}

        Random random = new Random(7);

        DateTime lastShuffleTime;

        Vector2 _windowSize;

        public bool isUpdatingState = false;

        public ShapeTestEnvironment(EnvironmentForm __form)
        {
            _Form = __form;

            tickEnvironment = F_tickEnvironment;

            lastShuffleTime = DateTime.Now;

            shapeObject = new GameObject(new Mechanics.Vector2(0, 0), new Mechanics.Vector2(200, 200));

            this.Host();
        }
        
        void shuffle()
        {
            float posX = random.NextSingle()*2-1;
            float posY = random.NextSingle()*2-1;
            float scale = random.NextSingle()*2-1;
            scale = Math.Max(100f, scale*250f);

            targetShape = SHAPES[random.NextInt64(0, SHAPES.Length)];

            switch (targetShape) {
                case "Square":
                    shapeObject.displayType = "RECTANGLE";
                break;
                case "Circle":
                    shapeObject.displayType = "CIRCLE";
                break;
                case "Triangle":
                    shapeObject.displayType = "TRIANGLE";
                break;
            }

            shapeObject.position = new Mechanics.Vector2(posX*300f, posY*200f);
            shapeObject.size = new Mechanics.Vector2(1, 1) * scale;
            lastShuffleTime = DateTime.Now;
        }

        public void displayEnvironment(Graphics graphics, Mechanics.Vector2 windowSize)
        {
            shapeObject.display(graphics, Brushes.White, windowSize);
            // if (idolizing)
            // {
            //     naisentObject.display(graphics, Brushes.Red, windowSize);
            // } else
            // {
            // }

            _windowSize = windowSize;
        }

        public bool F_tickEnvironment(float dt, HashSet<Keys> activeInputs, float passedDeltaTime)
        {
            if ((DateTime.Now-lastShuffleTime).TotalSeconds > .01f && (activeInputs.Contains(Keys.E) || true) && seenIt)
            {
                isUpdatingState = true;
                this.shuffle();
                seenIt = false;
                return true;
            }

            return false;
        }
    }
}