/*
 * The unit of distributed work: compute ONE band of rows of C = A x B on the member that
 * holds the band. The driver sends one BandTask to every member (submitToMember).
 *
 * Data layout (prepared by the driver, see DistMatMul):
 *   map mm-A, key keys[m] : rows [i0, i1) of A          (stored on member m)
 *   map mm-B, key keys[q] : rows [b[q], b[q+1]) of B    (chunk q stored on member q)
 *   map mm-C, key keys[m] : rows [i0, i1) of C          (written by this task, stored on member m)
 * keys[m] is chosen by the driver so that its Hazelcast partition is owned by member m.
 *
 * Steps on member m (each timed with System.nanoTime on the member):
 *   1. fetchA : read its own A band from the local partition and decode it
 *   2. fetchB : read ALL W chunks of B (its own chunk locally, the other W-1 over the
 *               network connection between members) and decode them into a full B
 *   3. compute: the blocked kernel of Assignment 3 (ParallelMatMul.multiplyBlockedRows,
 *               unchanged) on rows [i0, i1), split into `threads` contiguous sub-bands run
 *               by `threads` Java threads of this member (1 thread = run inline)
 *   4. storeC : encode the C band and write it to the local partition of mm-C
 * Each C[i][j] is computed by exactly one thread, with the products added in the same
 * k order as the sequential kernel, so the result is bitwise identical to it.
 *
 * The task returns one CSV fragment with its timings and byte counts. Whether every B
 * chunk was local or remote is checked from the partition owner, not assumed.
 */
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.HazelcastInstanceAware;
import com.hazelcast.map.IMap;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class BandTask implements Callable<String>, Serializable, HazelcastInstanceAware {

    private static final long serialVersionUID = 1L;
    static final int BS = ParallelMatMul.BS;   // 256, as in Assignments 2 and 3

    private final int n;
    private final int memberIndex;
    private final int i0;
    private final int i1;
    private final int[] chunkBounds;   // W + 1 row boundaries of the B chunks
    private final String[] keys;       // placement key of every member
    private final int threads;
    private transient HazelcastInstance hz;

    BandTask(int n, int memberIndex, int i0, int i1, int[] chunkBounds, String[] keys, int threads) {
        this.n = n;
        this.memberIndex = memberIndex;
        this.i0 = i0;
        this.i1 = i1;
        this.chunkBounds = chunkBounds;
        this.keys = keys;
        this.threads = threads;
    }

    @Override
    public void setHazelcastInstance(HazelcastInstance hazelcastInstance) {
        this.hz = hazelcastInstance;
    }

    // One fixed pool of compute threads per thread count, created on first use and reused
    // by every later task on this member (thread creation is then outside the timings).
    private static final Map<Integer, ExecutorService> POOLS = new HashMap<>();

    private static synchronized ExecutorService pool(int t) {
        return POOLS.computeIfAbsent(t, Executors::newFixedThreadPool);
    }

    static void computeBand(double[][] A, double[][] B, double[][] C, int n, int i0, int i1, int t) {
        if (t == 1) {
            ParallelMatMul.multiplyBlockedRows(A, B, C, n, BS, i0, i1);
            return;
        }
        List<Callable<Void>> parts = new ArrayList<>(t);
        int rows = i1 - i0;
        for (int q = 0; q < t; q++) {
            final int s0 = i0 + (int) ((long) rows * q / t);
            final int s1 = i0 + (int) ((long) rows * (q + 1) / t);
            parts.add(() -> {
                ParallelMatMul.multiplyBlockedRows(A, B, C, n, BS, s0, s1);
                return null;
            });
        }
        try {
            for (Future<Void> f : pool(t).invokeAll(parts)) {
                f.get();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String call() {
        IMap<String, byte[]> mapA = hz.getMap(ClusterConfig.MAP_A);
        IMap<String, byte[]> mapB = hz.getMap(ClusterConfig.MAP_B);
        IMap<String, byte[]> mapC = hz.getMap(ClusterConfig.MAP_C);
        int w = keys.length;

        long t0 = System.nanoTime();
        // 1. own band of A (local partition)
        double[][] A = new double[n][];   // only rows i0..i1-1 are allocated
        byte[] aBytes = mapA.get(keys[memberIndex]);
        RowCodec.decode(aBytes, A, i0, i1, n);
        long t1 = System.nanoTime();

        // 2. the whole of B: one chunk per member
        Set<String> all = new LinkedHashSet<>(java.util.Arrays.asList(keys));
        Map<String, byte[]> chunks = mapB.getAll(all);
        double[][] B = new double[n][];
        long localBytes = 0;
        long remoteBytes = 0;
        for (int q = 0; q < w; q++) {
            byte[] c = chunks.get(keys[q]);
            RowCodec.decode(c, B, chunkBounds[q], chunkBounds[q + 1], n);
            boolean local = hz.getPartitionService().getPartition(keys[q]).getOwner().localMember();
            if (local) {
                localBytes += c.length;
            } else {
                remoteBytes += c.length;
            }
        }
        long t2 = System.nanoTime();

        // 3. the kernel
        double[][] C = new double[n][];
        for (int i = i0; i < i1; i++) {
            C[i] = new double[n];
        }
        computeBand(A, B, C, n, i0, i1, threads);
        long t3 = System.nanoTime();

        // 4. store the C band on this member
        byte[] cBytes = RowCodec.encode(C, i0, i1, n);
        boolean cLocal = hz.getPartitionService().getPartition(keys[memberIndex]).getOwner().localMember();
        mapC.set(keys[memberIndex], cBytes);
        long t4 = System.nanoTime();

        return String.format(Locale.ROOT, "%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%d,%d,%d,%b,%s",
                memberIndex, i1 - i0, (t1 - t0) / 1e9, (t2 - t1) / 1e9, (t3 - t2) / 1e9,
                (t4 - t3) / 1e9, (t4 - t0) / 1e9, aBytes.length, localBytes, remoteBytes,
                cLocal, hz.getCluster().getLocalMember().getAddress().toString().replace(',', ';'));
    }

    // Untimed JIT pre-warm, as in Assignment 3: the kernel is called 1000 times on 32 x 32
    // matrices so that HotSpot compiles it as a normal method before the first band.
    static void preWarm() {
        int m = 32;
        java.util.Random rnd = new java.util.Random(1);
        double[][] a = new double[m][m];
        double[][] b = new double[m][m];
        double[][] c = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                a[i][j] = rnd.nextDouble();
                b[i][j] = rnd.nextDouble();
            }
        }
        for (int r = 0; r < 1000; r++) {
            ParallelMatMul.multiplyBlockedRows(a, b, c, m, BS, 0, m);
        }
    }
}
