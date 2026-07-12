using System;
using System.Drawing;
using System.Windows.Forms;

namespace Mechanics
{
    static class CONSTANTS
    {
        public const float GRAVITY = 150f;
        public const float y_VELOCITY_NUDGE_FACTOR = 100f;
        public const float x_VELOCITY_NUDGE_FACTOR = 1f;
    }

    public struct Vector2
    {
        public readonly float x;
        public readonly float y;

        public Vector2()
        {
            x = 0;
            y = 0;
        }
        public Vector2(float _x, float _y)
        {
            x = _x;
            y = _y;
        }

        public Vector2 nudgeTo(Vector2 other, Vector2 nudgeFactor)
        {
            float xDifference = other.x-this.x;
            float yDifference = other.y-this.y;

            float xNudge = other.x==this.x
            ? 0 : (xDifference/Math.Abs(xDifference))*nudgeFactor.x;
            float yNudge = other.y==this.y
            ? 0 : (yDifference/Math.Abs(yDifference))*nudgeFactor.y;

            return new Vector2(this.x+xNudge
                , this.y+yNudge);
        }

        public static Vector2 operator +(Vector2 v1, Vector2 v2)
        {
            return new Vector2(v1.x+v2.x, v1.y+v2.y);
        }

        public static Vector2 operator *(Vector2 v, float multiplier)
        {
            return new Vector2(v.x*multiplier, v.y*multiplier);
        }
        public static Vector2 operator /(Vector2 v, float divisor)
        {
            return new Vector2(v.x/divisor, v.y/divisor);
        }
    }

    public class GameObject
    {
        public Vector2 position;
        public Vector2 size;

        public string displayType;

        public GameObject(Vector2 _position, Vector2 _size)
        {
            this.position = _position;
            this.size = _size;
            this.displayType = "UNKNOWN";
        }

        public static GameObject Circle(Vector2 _position, float _radius)
        {
            GameObject gameObject = new GameObject(_position, new Vector2(_radius, _radius));
            gameObject.displayType = "CIRCLE";
            return gameObject;
        }
        public static GameObject Rectangle(Vector2 _position, Vector2 _size)
        {
            GameObject gameObject = new GameObject(_position, _size);
            gameObject.displayType = "RECTANGLE";
            return gameObject;
        }
        public static GameObject Triangle(Vector2 _position, Vector2 _size)
        {
            GameObject gameObject = new GameObject(_position, _size);
            gameObject.displayType = "TRIANGLE";
            return gameObject;
        }

        public void setPosition(Vector2 _position)
        {
            this.position = _position;
        }

        public void move(Vector2 _movement)
        {
            setPosition(this.position+_movement);
        }

        public void display(Graphics graphics, Brush color, Vector2 windowSize)
        {
            if (this.displayType == "CIRCLE")
            {
                // graphics.DrawString("Position: " + this.position.x + ", " + this.position.y,
                //     Control.DefaultFont, Brushes.Yellow, 50, 250);
                // graphics.FillEllipse(Brushes.Blue, 100f+(windowSize.x/2), (-100f)+(windowSize.y/2), this.size.x, this.size.y);
                graphics.FillEllipse(color
                , (this.position.x-(this.size.x/2))+(windowSize.x/2)
                , (-this.position.y-(this.size.y/2))+(windowSize.y/2)

                , this.size.x, this.size.y);
            }
            else if (this.displayType == "RECTANGLE")
            {
                // graphics.DrawString("Position: " + this.position.x + ", " + this.position.y,
                //     Control.DefaultFont, Brushes.Yellow, 50, 500);
                // graphics.FillEllipse(Brushes.Red, 100f, 150f, 40, 40);
                graphics.FillRectangle(color
                , (this.position.x-(this.size.x/2))+(windowSize.x/2)
                , (-this.position.y-(this.size.y/2))+(windowSize.y/2)

                , this.size.x, this.size.y);
            }
            else if (this.displayType == "TRIANGLE")
            {
                // graphics.DrawString("Position: " + this.position.x + ", " + this.position.y,
                //     Control.DefaultFont, Brushes.Yellow, 50, 500);
                // graphics.FillEllipse(Brushes.Red, 100f, 150f, 40, 40);

                float cornerX = (this.position.x-(this.size.x/2))+(windowSize.x/2);
                float cornerY = (-this.position.y-(this.size.y/2))+(windowSize.y/2);

                Point[] points = {
                    new Point((int)(cornerX+(this.size.x/2)), (int)(cornerY)),
                    new Point((int)(cornerX), (int)(cornerY+this.size.y)),
                    new Point((int)(cornerX+this.size.x), (int)(cornerY+this.size.y))
                };
                
                graphics.FillPolygon(color, points);
            }
        }
    }
    public class PhysicalGameObject
    {
        public Vector2 velocity {get; set;}
        public GameObject gameObject {get; private set;}

        public PhysicalGameObject(GameObject _gameObject)
        {
            this.velocity = new Vector2();
            this.gameObject = _gameObject;
        }

        public void addVelocity(Vector2 _velocity)
        {
            this.velocity += _velocity;
        }

        public void tickVelocity(float dt
        , float positiveXBoundary, float negativeXBoundary
        , float positiveYBoundary, float negativeYBoundary)
        {
            this.gameObject.move(velocity*dt);

            Vector2 afterBoundaryAdjustments = this.gameObject.position;

            Vector2 distanceSize = this.gameObject.size/2;

            if (this.gameObject.position.x+distanceSize.x > positiveXBoundary)
            {
                afterBoundaryAdjustments = new Vector2(positiveXBoundary-distanceSize.x, afterBoundaryAdjustments.y);
            }
            else if (this.gameObject.position.x-distanceSize.x < negativeXBoundary)
            {
                afterBoundaryAdjustments = new Vector2(negativeXBoundary+distanceSize.x, afterBoundaryAdjustments.y);
            }
            
            if (this.gameObject.position.y+distanceSize.y > positiveYBoundary)
            {
                afterBoundaryAdjustments = new Vector2(afterBoundaryAdjustments.x, positiveYBoundary-distanceSize.y);
            }
            else if (this.gameObject.position.y-distanceSize.y < negativeYBoundary)
            {
                afterBoundaryAdjustments = new Vector2(afterBoundaryAdjustments.x, negativeYBoundary+distanceSize.y);
            }

            this.gameObject.setPosition(afterBoundaryAdjustments);
            velocity = velocity.nudgeTo(new Vector2(0, -CONSTANTS.GRAVITY)
            , new Vector2(CONSTANTS.x_VELOCITY_NUDGE_FACTOR*dt
            , CONSTANTS.y_VELOCITY_NUDGE_FACTOR*dt));
        }

        public bool isColliding(GameObject other)
        {
            bool xIsColliding = ((this.gameObject.position.x+this.gameObject.size.x/2) >= (other.position.x-other.size.x/2))
            && ((this.gameObject.position.x-this.gameObject.size.x/2) <= (other.position.x+other.size.x/2));
            bool yIsColliding = ((this.gameObject.position.y+this.gameObject.size.y/2) >= (other.position.y-other.size.y/2))
            && ((this.gameObject.position.y-this.gameObject.size.y/2) <= (other.position.y+other.size.y/2));

            return xIsColliding && yIsColliding;
        }
    }
}