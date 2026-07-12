using Display_WinForms_;
using Environments;
using Extras;

namespace Guides
{
    public static class ShapeTest_AbsoluteSupervision_Guide
    {
        public static float[] Respond(ShapeTestEnvironment environment)
        {
            float[] desiredAction = [0, 0, 0];

            if (environment.targetShape == "Square")
            {
                // desiredMoveDirection = 0; case 2:
                desiredAction[2] = 1;
            }
            else if (environment.targetShape == "Triangle")
            {
                // desiredMoveDirection = 1; case 1:
                desiredAction[1] = 1;
            }
            else if (environment.targetShape == "Circle")
            {
                // desiredMoveDirection = -1; case 0:
                desiredAction[0] = 1;
            }

            return desiredAction;
        }
    }
}