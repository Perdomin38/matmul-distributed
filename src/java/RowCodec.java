/*
 * Conversion between a block of matrix rows and the byte[] stored in a Hazelcast map.
 * Rows r0 <= i < r1 of an n-column matrix become (r1 - r0) * n little-endian float64
 * values, row-major: 8 * (r1 - r0) * n bytes, with no header. Hazelcast serialises a
 * byte[] value as the bytes themselves plus a few bytes of type and length, so the
 * application payload of every transfer is known exactly.
 */
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

final class RowCodec {

    private RowCodec() {
    }

    static byte[] encode(double[][] m, int r0, int r1, int n) {
        ByteBuffer buf = ByteBuffer.allocate(8 * (r1 - r0) * n).order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer d = buf.asDoubleBuffer();
        for (int i = r0; i < r1; i++) {
            d.put(m[i], 0, n);
        }
        return buf.array();
    }

    /** Decodes the bytes into rows r0..r1-1 of m, allocating each row. */
    static void decode(byte[] bytes, double[][] m, int r0, int r1, int n) {
        if (bytes.length != 8L * (r1 - r0) * n) {
            throw new IllegalStateException("expected " + 8L * (r1 - r0) * n + " bytes, got " + bytes.length);
        }
        DoubleBuffer d = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        for (int i = r0; i < r1; i++) {
            double[] row = new double[n];
            d.get(row, 0, n);
            m[i] = row;
        }
    }

    /** Decodes into rows that already exist (used by the driver for C). */
    static void decodeInto(byte[] bytes, double[][] m, int r0, int r1, int n) {
        if (bytes.length != 8L * (r1 - r0) * n) {
            throw new IllegalStateException("expected " + 8L * (r1 - r0) * n + " bytes, got " + bytes.length);
        }
        DoubleBuffer d = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        for (int i = r0; i < r1; i++) {
            d.get(m[i], 0, n);
        }
    }
}
