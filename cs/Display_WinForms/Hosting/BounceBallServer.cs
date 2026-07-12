using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

using Guides;

namespace Environments
{
    partial class BounceBallEnvironment
    {
        bool hadRestartedPending = false;

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

            // byte[] buffer = new byte[4096];

            // new BounceBallState(this).asVisualInput();

            byte start = SharedUtil.ReadByte(stream);
            if (start <= 0) return;

            while (true)
            {
                byte want = SharedUtil.ReadByte(stream);
                if (want <= 0) continue;

                if (this.hadRestartedPending)
                {
                    this.hadRestarted = false;
                    this.hadRestartedPending = false;
                }
                if (this.hadRestarted)
                {
                    this.hadRestartedPending = true;
                }

                if (hadRestarted)
                {
                    stream.WriteByte((byte)1);
                } else
                {
                    stream.WriteByte((byte)0);
                }
                
                BounceBallState state;
                lock (stateLockMutex) {
                    state = new BounceBallState(this);
                }
                
                float[] visualInput = state.asVisualInput();
                _Form.smallPreview.SetBitmap(_Form.downSampler.tensorView);
                
                try
                {
                    SharedUtil.WriteFloatArray(stream, visualInput);
                } catch (Exception e)
                {
                    Console.WriteLine($"Excpetion while sending visual input (most likely disconnected): {e.ToString()}");
                    Console.WriteLine("Exiting BounceBall!");
                    Application.Exit();
                    return;
                }

                float[] actionValues = SharedUtil.ReadFloatArray(stream);

                int finalAction;

                float[] desiredAction = BounceBall_AbsoluteSupervision_Guide.Respond(state);
                
                if (!idolizing)
                {
                    finalAction = getHighestNumberIndex(actionValues);
                } else
                {
                    finalAction = getHighestNumberIndex(desiredAction);
                }

                int decidedMoveDirection = 0;
                switch(finalAction)
                {
                    case 0:
                        decidedMoveDirection = -1;
                        break;
                    case 1:
                        decidedMoveDirection = 1;
                        break;
                    case 2:
                        decidedMoveDirection = 0;
                        break;
                }

                lock(stateLockMutex) {
                    moveDirection = decidedMoveDirection;
                }

                SharedUtil.WriteFloatArray(stream, desiredAction);
                
                byte procceed = SharedUtil.ReadByte(stream);
                if (procceed <= 0) return;
            }
        }

        private void Host()
        {
            new Thread(IPCServer) { IsBackground = true }.Start();
        }
    }
}