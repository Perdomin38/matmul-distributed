/*
 * DRIVER of the distributed matrix multiplication C = A x B (n x n, float64).
 *
 * The driver is a Hazelcast CLIENT in its own JVM: it holds no map data and runs no band
 * task, so all computation happens in the W worker members (DistMember processes).
 *
 * Partitioning (W = number of members, member m = 0..W-1 in join order):
 *   - A and C: W contiguous bands of rows, [floor(n m / W), floor(n (m+1) / W)), as in the
 *     static partition of Assignment 3.
 *   - B: W contiguous chunks of rows with the same boundaries; every task needs ALL of B.
 *   - Placement: for each member m the driver finds a key keys[m] whose Hazelcast
 *     partition is owned by member m. Band m of A, chunk m of B and band m of C are all
 *     stored under keys[m], so they live on member m, and the band task m is sent to
 *     member m (submitToMember). Maps have no backups (ClusterConfig).
 *
 * One end-to-end multiplication (the timed region, System.nanoTime in the driver), made
 * of three SEQUENTIAL phases whose times add up to the total:
 *   distribute : encode the A bands and B chunks to byte[] and putAll them into mm-A and
 *                mm-B (driver -> members: 8n^2 + 8n^2 bytes of payload)
 *   execute    : submit the W band tasks and wait for all of them. Inside, every member
 *                fetches B (member <-> member: the W-1 chunks it does not hold),
 *                computes its band and stores its C band locally. The task phases of
 *                different members run CONCURRENTLY, so they are reported per task and
 *                never added across members.
 *   collect    : getAll of the W C bands from mm-C and decode them into the driver's C
 *                (members -> driver: 8n^2 bytes)
 * Outside the timed region: reading the input files, allocating C, connecting to the
 * cluster, finding the placement keys, clearing the maps before every run, and the
 * checksum of C (which consumes the result so the work cannot be skipped).
 *
 * Usage: java DistMatMul <clusterName> <members W> <threadsPerMember t> <n> <A.bin> <B.bin>
 *                        <warmups> <reps> [C_out.bin]
 * Output (stdout), CSV lines of three kinds:
 *   setup,<connect_s>,<cluster_wait_s>,<ready_epoch_ms>
 *   run,<phase>,<run>,<total_s>,<distribute_s>,<execute_s>,<collect_s>,
 *       <bytes_to_cluster>,<bytes_from_cluster>,<checksum>
 *   task,<phase>,<run>,<member>,<rows>,<fetchA_s>,<fetchB_s>,<compute_s>,<storeC_s>,
 *       <task_s>,<bytesA>,<bytesB_local>,<bytesB_remote>,<C_local>,<address>
 */
import com.hazelcast.client.HazelcastClient;
import com.hazelcast.cluster.Member;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.IExecutorService;
import com.hazelcast.map.IMap;
import com.hazelcast.partition.Partition;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Future;

public class DistMatMul {

    static String placementKey(HazelcastInstance hz, Member owner, int m) {
        for (int c = 0; c < 1_000_000; c++) {
            String key = "band-" + m + "-" + c;
            Partition p = hz.getPartitionService().getPartition(key);
            Member o = p.getOwner();
            if (o != null && o.getUuid().equals(owner.getUuid())) {
                return key;
            }
        }
        throw new IllegalStateException("no key found for member " + m);
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 8) {
            System.err.println("usage: java DistMatMul cluster W t n A.bin B.bin warmups reps [C_out.bin]");
            System.exit(1);
        }
        String cluster = args[0];
        int w = Integer.parseInt(args[1]);
        int t = Integer.parseInt(args[2]);
        int n = Integer.parseInt(args[3]);
        int warmups = Integer.parseInt(args[6]);
        int reps = Integer.parseInt(args[7]);

        double[][] A = ParallelMatMul.readMatrix(args[4], n);
        double[][] B = ParallelMatMul.readMatrix(args[5], n);
        double[][] C = new double[n][n];

        // ---- setup (not part of the per-multiplication time) --------------------------
        long s0 = System.nanoTime();
        HazelcastInstance hz = HazelcastClient.newHazelcastClient(ClusterConfig.client(cluster));
        long s1 = System.nanoTime();
        long deadline = System.currentTimeMillis() + 180_000;
        while (hz.getCluster().getMembers().size() < w) {
            if (System.currentTimeMillis() > deadline) {
                throw new IllegalStateException("only " + hz.getCluster().getMembers().size()
                        + " of " + w + " members joined");
            }
            Thread.sleep(50);
        }
        if (hz.getCluster().getMembers().size() != w) {
            throw new IllegalStateException("cluster has " + hz.getCluster().getMembers().size()
                    + " members, expected " + w);
        }
        List<Member> members = new ArrayList<>(hz.getCluster().getMembers());
        // wait until the partition table assigns an owner to every partition
        while (true) {
            boolean ok = true;
            for (Partition p : hz.getPartitionService().getPartitions()) {
                if (p.getOwner() == null) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                break;
            }
            Thread.sleep(50);
        }
        long s2 = System.nanoTime();
        System.out.printf(Locale.ROOT, "setup,%.6f,%.6f,%d%n", (s1 - s0) / 1e9, (s2 - s1) / 1e9,
                System.currentTimeMillis());

        int[] bounds = new int[w + 1];
        for (int m = 0; m <= w; m++) {
            bounds[m] = (int) ((long) n * m / w);
        }
        String[] keys = new String[w];
        for (int m = 0; m < w; m++) {
            keys[m] = placementKey(hz, members.get(m), m);
            System.err.println("placement member " + m + " " + members.get(m).getAddress()
                    + " key=" + keys[m] + " rows=[" + bounds[m] + "," + bounds[m + 1] + ")");
        }
        Set<String> keySet = new LinkedHashSet<>(java.util.Arrays.asList(keys));

        IMap<String, byte[]> mapA = hz.getMap(ClusterConfig.MAP_A);
        IMap<String, byte[]> mapB = hz.getMap(ClusterConfig.MAP_B);
        IMap<String, byte[]> mapC = hz.getMap(ClusterConfig.MAP_C);
        IExecutorService exec = hz.getExecutorService(ClusterConfig.EXECUTOR);

        for (int r = 0; r < warmups + reps; r++) {
            boolean isWarmup = r < warmups;
            String phase = isWarmup ? "warmup" : "measured";
            int run = isWarmup ? r + 1 : r - warmups + 1;
            mapA.clear();
            mapB.clear();
            mapC.clear();
            for (int i = 0; i < n; i++) {
                java.util.Arrays.fill(C[i], Double.NaN);   // a missing band would show up
            }

            long t0 = System.nanoTime();
            // distribute
            Map<String, byte[]> aParts = new HashMap<>();
            Map<String, byte[]> bParts = new HashMap<>();
            long toCluster = 0;
            for (int m = 0; m < w; m++) {
                byte[] a = RowCodec.encode(A, bounds[m], bounds[m + 1], n);
                byte[] b = RowCodec.encode(B, bounds[m], bounds[m + 1], n);
                aParts.put(keys[m], a);
                bParts.put(keys[m], b);
                toCluster += a.length + b.length;
            }
            mapA.putAll(aParts);
            mapB.putAll(bParts);
            aParts = null;
            bParts = null;
            long t1 = System.nanoTime();

            // execute
            List<Future<String>> futures = new ArrayList<>(w);
            for (int m = 0; m < w; m++) {
                futures.add(exec.submitToMember(
                        new BandTask(n, m, bounds[m], bounds[m + 1], bounds, keys, t), members.get(m)));
            }
            List<String> taskLines = new ArrayList<>(w);
            for (Future<String> f : futures) {
                taskLines.add(f.get());
            }
            long t2 = System.nanoTime();

            // collect
            Map<String, byte[]> cParts = mapC.getAll(keySet);
            long fromCluster = 0;
            for (int m = 0; m < w; m++) {
                byte[] c = cParts.get(keys[m]);
                RowCodec.decodeInto(c, C, bounds[m], bounds[m + 1], n);
                fromCluster += c.length;
            }
            long t3 = System.nanoTime();

            double checksum = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    checksum += C[i][j];
                }
            }
            System.out.printf(Locale.ROOT, "run,%s,%d,%.9f,%.9f,%.9f,%.9f,%d,%d,%s%n", phase, run,
                    (t3 - t0) / 1e9, (t1 - t0) / 1e9, (t2 - t1) / 1e9, (t3 - t2) / 1e9,
                    toCluster, fromCluster, Double.toString(checksum));
            for (String line : taskLines) {
                System.out.println("task," + phase + "," + run + "," + line);
            }
            System.out.flush();
        }

        if (args.length >= 9) {
            ParallelMatMul.writeMatrix(args[8], C, n);
        }
        mapA.clear();
        mapB.clear();
        mapC.clear();
        hz.shutdown();
    }
}
