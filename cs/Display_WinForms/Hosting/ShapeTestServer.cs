using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

using Guides;

namespace Environments
{
    partial class ShapeTestEnvironment
    {
        public bool seenIt = true;

        private int getHighestNumberIndex(float[] numbers)
        {
            int highestIndex = 0;
            float highestNumber = -float.MaxValue;
            for (int i = 0; i < numbers.Count(); i++)
            {
                if (numbers[i] > highestNumber)
                {
                    highestIndex = i;
                    highestNumber = numbers[i];
                }
            }
            return highestIndex;
        }

        private void IPCServer()
        {
            TcpListener listener = new TcpListener(IPAddress.Loopback, 5555);
            listener.Start();

            using TcpClient client = listener.AcceptTcpClient();
            using NetworkStream stream = client.GetStream();

            byte start = SharedUtil.ReadByte(stream);
            if (start == 0) return;

            while (true)
            {
                while (true) {
                    if (!seenIt)
                    {
                        break;
                    }
                }
                while (true) {
                    if (!isUpdatingState)
                    {
                        break;
                    }
                }

                float[] visualInput = _Form.downSampler.tensor;
                _Form.smallPreview.SetBitmap(_Form.downSampler.tensorView);
                float[] desiredAction = ShapeTest_AbsoluteSupervision_Guide.Respond(this);
                
                SharedUtil.WriteFloatArray(stream, visualInput);

                byte getDesired = SharedUtil.ReadByte(stream);
                if (getDesired == 0) return;

                // float[] actionValues = SharedUtil.ReadFloatArray(stream);

                SharedUtil.WriteFloatArray(stream, desiredAction);
                
                seenIt = true;

                byte procceed = SharedUtil.ReadByte(stream);
                if (procceed == 0) return;
            }
        }

        private void Host()
        {
            new Thread(IPCServer) { IsBackground = true }.Start();
        }
    }
}