using System.Net.Sockets;

namespace Environments
{
    public static class SharedUtil
    {
        public static void ReadExactly(Stream stream, Span<byte> buffer)
        {
            int pos = 0;
            int remaining = buffer.Length;

            while (remaining > 0)
            {
                int read = stream.Read(buffer.Slice(pos, remaining));
                if (read == 0)
                {
                    // Remote side closed the connection
                    throw new EndOfStreamException("Stream closed before reading enough bytes.");
                }

                pos += read;
                remaining -= read;
            }
        }
        public static void WriteInt(NetworkStream s, int v)
        {
            var b = BitConverter.GetBytes(v);
            s.Write(b, 0, 4);
        }

        public static void WriteFloat(NetworkStream s, float v)
        {
            var b = BitConverter.GetBytes(v);
            s.Write(b, 0, 4);
        }

        public static int ReadInt(NetworkStream s)
        {
            Span<byte> b = stackalloc byte[4];
            ReadExactly(s, b);
            return BitConverter.ToInt32(b);
        }

        public static byte ReadByte(NetworkStream s)
        {
            int v = s.ReadByte();
            if (v < 0) throw new IOException("Disconnected");
            return (byte)v;
        }

        public static void WriteFloatArray(NetworkStream s, float[] data)
        {
            // prefix length
            WriteInt(s, data.Length);
            
            byte[] buf = new byte[data.Length * 4];
            Buffer.BlockCopy(data, 0, buf, 0, buf.Length);
            s.Write(buf, 0, buf.Length);
        }

        public static float[] ReadFloatArray(NetworkStream s)
        {
            int len = ReadInt(s);
            float[] data = new float[len];
            byte[] buf = new byte[len * 4];
            ReadExactly(s, buf);
            Buffer.BlockCopy(buf, 0, data, 0, buf.Length);
            return data;
        }

    }
}