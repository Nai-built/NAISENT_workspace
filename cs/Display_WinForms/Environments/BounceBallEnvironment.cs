using Mechanics;
using Display_WinForms_;

namespace Environments
{
    public delegate bool EnvironmentTick(float dt, HashSet<Keys> activeInputs, float passedDeltaTime);

    public partial class BounceBallEnvironment
    {
        public EnvironmentForm _Form {get; private set;}

        static float MAX_SESSION_TIME = 5f;

        public PhysicalGameObject ball {get; private set;}
        public GameObject naisentObject {get; private set;}

        Vector2 _windowSize;

        public DateTime startTime;
        DateTime appStartedTime;

        float lastBounceTime = 0;

        public int moveDirection {get; private set;} = 0;

        private object stateLockMutex = new object();

        bool isHosting = true;

        bool hadRestarted = true;

        bool idolizing = true;

        public EnvironmentTick tickEnvironment {get; private set;}

        Random random = new Random(0);

        Vector2 initializeBallPosition()
        {
            float rValue = random.NextSingle();
            rValue = (rValue*2)-1;
            return new Vector2(rValue*200, 100);
        }

        public BounceBallEnvironment(EnvironmentForm __form)
        {
            this._Form = __form;

            ball = new PhysicalGameObject(GameObject.Circle(initializeBallPosition(), 50));
            naisentObject = GameObject.Rectangle(new Vector2(0, -200), new Vector2(80, 20));

            tickEnvironment = F_tickEnvironment;

            if (isHosting)
            {
                this.Host();
            }

            this.startTime = DateTime.Now;
            this.appStartedTime = DateTime.Now;
        }

        public void displayEnvironment(Graphics graphics, Vector2 windowSize)
        {
            ball.gameObject.display(graphics, Brushes.Blue, windowSize);
            naisentObject.display(graphics, Brushes.White, windowSize);
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
            if (hadRestarted)
            {
                return true;
            }

            ball.tickVelocity(dt
                , _windowSize.x/2, -_windowSize.x/2
                , _windowSize.y/2, -_windowSize.y/2);

            if (!isHosting)
            {
                if (activeInputs.Contains(Keys.D)) {
                    lock(stateLockMutex) {
                        moveDirection = 1;
                    }
                }
                else if (activeInputs.Contains(Keys.A)) {
                    lock(stateLockMutex) {
                        moveDirection = -1;
                    }
                }
                else {
                    lock(stateLockMutex) {
                        moveDirection = 0;
                    }
                }
            }
            naisentObject.move(new Vector2(150f*moveDirection, 0)*dt);

            if (ball.isColliding(naisentObject) && passedDeltaTime > lastBounceTime+.5f)
            {
                lastBounceTime = passedDeltaTime;

                float xDifference = ball.gameObject.position.x-naisentObject.position.x;
                float xDirection = ball.gameObject.position.x == naisentObject.position.x ? 0
                    : (xDifference/naisentObject.size.x);
                // float xDirection = (random.NextSingle()*2)-1;
                ball.addVelocity(new Vector2(xDirection*100, 400));
            }

            if (ball.gameObject.position.x + (ball.gameObject.size.x/2) >= _windowSize.x/2)
            {
                ball.addVelocity(new Vector2(-100, 0));
            }
            else if (ball.gameObject.position.x - (ball.gameObject.size.x/2) <= -_windowSize.x/2)
            {
                ball.addVelocity(new Vector2(100, 0));
            }

            if (naisentObject.position.x > _windowSize.x/2)
            {
                naisentObject.setPosition(
                    new Vector2(_windowSize.x/2, naisentObject.position.y));
            }
            else if (naisentObject.position.x < -_windowSize.x/2)
            {
                naisentObject.setPosition(
                    new Vector2(-_windowSize.x/2, naisentObject.position.y));
            }

            TimeSpan timePassed = DateTime.Now-this.startTime;
            if (ball.gameObject.position.y < -250 || timePassed.TotalSeconds > MAX_SESSION_TIME)
            {
                this.hadRestarted = true;

                this.moveDirection = 0;

                ball.velocity = new Vector2();
                ball.gameObject.setPosition(initializeBallPosition());
                naisentObject.setPosition(new Vector2(0, -200));

                this.startTime = DateTime.Now;
            }
            if (ball.gameObject.position.y >= 50)
            {
                idolizing = false;
            } else
            {
                idolizing = true;
            }
            
            TimeSpan appTimePassed = DateTime.Now-this.appStartedTime;
            if (appTimePassed.TotalMinutes > 2)
            {
                idolizing = false;
            }

            return true;
        }
    }
}