using Display_WinForms_;
using Environments;
using Extras;

namespace Guides
{
    public struct BounceBallState
    {
        public readonly float ballPosX;
        public readonly float ballPosY;
        public readonly float ballVelX;
        public readonly float ballVelY;

        public readonly float naisentObjectPosX;
        public readonly int naisentObjectMoveDirection;

        private float[] tensor;

        public BounceBallState(BounceBallEnvironment environment)
        {
            this.ballPosX = environment.ball.gameObject.position.x;
            this.ballPosY = environment.ball.gameObject.position.y;
            this.ballVelX = environment.ball.velocity.x;
            this.ballVelY = environment.ball.velocity.y;

            this.naisentObjectPosX = environment.naisentObject.position.x;
            this.naisentObjectMoveDirection = environment.moveDirection;

            this.tensor = environment._Form.downSampler.tensor;
        }

        public object toObject()
        {
            return new
            {
                ballPosX = this.ballPosX,
                ballPosY = this.ballPosY,
                ballVelX = this.ballVelX,
                ballVelY = this.ballVelY,

                naisentObjectPosX = this.naisentObjectPosX,
                naisentObjectMoveDirection = this.naisentObjectMoveDirection,
            };
        }

        public object asInput()
        {
            return new float[] {this.ballPosX, this.ballPosY, this.naisentObjectPosX};
        }

        public float[] asVisualInput()
        {
            return tensor;
        }
    }

    public static class BounceBall_AbsoluteSupervision_Guide
    {
        public static float[] Respond(BounceBallState state)
        {
            float[] desiredAction = [0, 0, 0];

            float xDifference = (state.ballPosX + (state.ballVelX*0.0f))-state.naisentObjectPosX;
            if (Math.Abs(xDifference) <= 10f)
            {
                // desiredMoveDirection = 0; case 2:
                desiredAction[2] = 1;
            }
            else if (xDifference > 0)
            {
                // desiredMoveDirection = 1; case 1:
                desiredAction[1] = 1;
            }
            else if (xDifference < 0)
            {
                // desiredMoveDirection = -1; case 0:
                desiredAction[0] = 1;
            }

            return desiredAction;
        }
    }
}