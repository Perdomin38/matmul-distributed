/*
 * Shared-memory parallel matrix multiplication C = A x B (n x n, float64) in Java,
 * and its equivalent sequential baseline, run by the SAME timing harness.
 *
 * Storage: double[n][n] (array of row arrays), as in Assignments 1 and 2.
 *
 * Methods
 *   seq   sequential baseline: the cache-blocked i-k-j kernel with 256 x 256 tiles,
 *         copied unchanged from Assignment 2 (DenseMatMul.multiplyBlocked, repository
 *         matmul-optimised-sparse), the fastest dense method of that study.
 *   par   the same blocked kernel, parallelised with p Java platform threads. The rows
 *         of C are split into p contiguous bands of floor(n/p) or ceil(n/p) rows
 *         (static partition, like OpenMP schedule(static)); thread t runs the blocked
 *         kernel on its band only. Each C[i][j] is still computed by one thread with
 *         the products added in the same k order, so the result is bitwise identical
 *         to seq.
 *
 * Shared data: A and B are only read. Every thread writes only the rows of C in its
 * own band, so no two threads write the same element and no locks are needed.
 * invokeAll() returns only after every task has finished, and Future.get() gives the
 * main thread a happens-before edge with each task, so the main thread sees all of C.
 *
 * Usage: java ParallelMatMul <seq|par> <threads> <n> <A.bin> <B.bin> <warmups> <reps> [C_out.bin]
 *   Input files: n*n little-endian float64 values, row-major.
 *   Output (stdout): CSV lines "phase,run,calls,time_s,checksum"
 *     time_s is the time of ONE multiplication; if one multiplication takes less than
 *     MIN_BATCH_S, every run executes `calls` multiplications back to back and time_s
 *     is the batch time divided by `calls`.
 *
 * Timed region: the multiplication only (zeroing C, creating the p tasks, running them
 * and waiting for all of them). Reading the inputs, allocating C and creating the
 * thread pool happen before the timer starts; the pool threads are started during
 * the untimed pre-warm, so thread creation is not timed.
 */
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class ParallelMatMul {

    static final int BS = 256;              // block size chosen by the Assignment 2 pilot
    static final double MIN_BATCH_S = 0.05; // shorter multiplications are timed in batches

    static double[][] readMatrix(String path, int n) throws IOException {
        ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(Paths.get(path)))
                .order(ByteOrder.LITTLE_ENDIAN);
        if (buf.capacity() != 8L * n * n) {
            throw new IOException("file " + path + " does not contain " + (n * n) + " values");
        }
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] = buf.getDouble();
            }
        }
        return m;
    }

    static void writeMatrix(String path, double[][] m, int n) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 * n * n).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                buf.putDouble(m[i][j]);
            }
        }
        Files.write(Paths.get(path), buf.array());
    }

    // Sequential baseline: identical to DenseMatMul.multiplyBlocked of Assignment 2.
    static void multiplyBlocked(double[][] A, double[][] B, double[][] C, int n, int bs) {
        for (int i = 0; i < n; i++) {
            Arrays.fill(C[i], 0.0);
        }
        for (int ii = 0; ii < n; ii += bs) {
            int iEnd = Math.min(ii + bs, n);
            for (int kk = 0; kk < n; kk += bs) {
                int kEnd = Math.min(kk + bs, n);
                for (int jj = 0; jj < n; jj += bs) {
                    int jEnd = Math.min(jj + bs, n);
                    for (int i = ii; i < iEnd; i++) {
                        double[] Ci = C[i];
                        double[] Ai = A[i];
                        for (int k = kk; k < kEnd; k++) {
                            double aik = Ai[k];
                            double[] Bk = B[k];
                            for (int j = jj; j < jEnd; j++) {
                                Ci[j] += aik * Bk[j];
                            }
                        }
                    }
                }
            }
        }
    }

    // The same kernel restricted to the rows i0 <= i < i1 (one thread's band).
    // With i0 = 0 and i1 = n it performs exactly the loops of multiplyBlocked.
    static void multiplyBlockedRows(double[][] A, double[][] B, double[][] C, int n, int bs,
                                    int i0, int i1) {
        for (int i = i0; i < i1; i++) {
            Arrays.fill(C[i], 0.0);
        }
        for (int ii = i0; ii < i1; ii += bs) {
            int iEnd = Math.min(ii + bs, i1);
            for (int kk = 0; kk < n; kk += bs) {
                int kEnd = Math.min(kk + bs, n);
                for (int jj = 0; jj < n; jj += bs) {
                    int jEnd = Math.min(jj + bs, n);
                    for (int i = ii; i < iEnd; i++) {
                        double[] Ci = C[i];
                        double[] Ai = A[i];
                        for (int k = kk; k < kEnd; k++) {
                            double aik = Ai[k];
                            double[] Bk = B[k];
                            for (int j = jj; j < jEnd; j++) {
                                Ci[j] += aik * Bk[j];
                            }
                        }
                    }
                }
            }
        }
    }

    static void multiplyParallel(double[][] A, double[][] B, double[][] C, int n, int bs,
                                 ExecutorService pool, int p) {
        List<Callable<Void>> tasks = new ArrayList<>(p);
        for (int t = 0; t < p; t++) {
            final int i0 = (int) ((long) n * t / p);
            final int i1 = (int) ((long) n * (t + 1) / p);
            final int id = t;
            tasks.add(() -> {
                long t0 = THREAD_TIMES ? System.nanoTime() : 0L;
                multiplyBlockedRows(A, B, C, n, bs, i0, i1);
                if (THREAD_TIMES) {
                    threadTime[id] = System.nanoTime() - t0;
                }
                return null;
            });
        }
        try {
            for (Future<Void> f : pool.invokeAll(tasks)) {
                f.get();   // rethrows any exception of a task
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    // Diagnostic (off in the main benchmark): -Dmatmul.threadTimes=true makes every task
    // record its own elapsed time, printed to stderr after each run as
    // "threadtimes,<phase>,<run>,<t_0 s>,...,<t_p-1 s>". Used by supplementary.py to
    // measure load imbalance between the threads.
    static final boolean THREAD_TIMES = Boolean.getBoolean("matmul.threadTimes");
    static long[] threadTime;

    static String method;
    static int threads;
    static ExecutorService pool;

    static void multiply(double[][] A, double[][] B, double[][] C, int n) {
        if (method.equals("seq")) {
            multiplyBlocked(A, B, C, n, BS);
        } else {
            multiplyParallel(A, B, C, n, BS, pool, threads);
        }
    }

    // JIT pre-warm (not timed), as in Assignment 2: the kernel is called 1000 times on
    // 32 x 32 matrices so that HotSpot compiles it as a normal method (not slower OSR
    // code) before the first large run. For par it also starts the p pool threads.
    static void preWarm() {
        int m = 32;
        java.util.Random rnd = new java.util.Random(1);
        double[][] a = new double[m][m], b = new double[m][m], c = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                a[i][j] = rnd.nextDouble();
                b[i][j] = rnd.nextDouble();
            }
        }
        for (int r = 0; r < 1000; r++) {
            multiply(a, b, c, m);
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 7) {
            System.err.println("usage: java ParallelMatMul seq|par threads n A.bin B.bin warmups reps [C_out.bin]");
            System.exit(1);
        }
        method = args[0];
        threads = Integer.parseInt(args[1]);
        int n = Integer.parseInt(args[2]);
        int warmups = Integer.parseInt(args[5]);
        int reps = Integer.parseInt(args[6]);
        if (!method.equals("seq") && !method.equals("par")) {
            throw new IllegalArgumentException("unknown method " + method);
        }
        if (method.equals("seq") && threads != 1) {
            throw new IllegalArgumentException("seq runs with exactly 1 thread");
        }

        double[][] A = readMatrix(args[3], n);
        double[][] B = readMatrix(args[4], n);
        double[][] C = new double[n][n];
        if (method.equals("par")) {
            pool = Executors.newFixedThreadPool(threads);
            threadTime = new long[threads];
        }
        System.err.println("availableProcessors=" + Runtime.getRuntime().availableProcessors());

        if (warmups > 0) {
            preWarm();
        }
        // Batch size for very short multiplications, decided from the last warm-up.
        int calls = 1;
        System.out.println("phase,run,calls,time_s,checksum");
        for (int r = 0; r < warmups + reps; r++) {
            boolean isWarmup = r < warmups;
            int c = isWarmup ? 1 : calls;
            long start = System.nanoTime();
            for (int q = 0; q < c; q++) {
                multiply(A, B, C, n);
            }
            double perCall = (System.nanoTime() - start) / 1e9 / c;
            if (r == warmups - 1 && perCall < MIN_BATCH_S) {
                calls = (int) Math.ceil(MIN_BATCH_S / perCall);
            }

            // The checksum consumes the result, so the work cannot be skipped.
            double checksum = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    checksum += C[i][j];
                }
            }
            System.out.printf(java.util.Locale.ROOT, "%s,%d,%d,%.9f,%s%n",
                    isWarmup ? "warmup" : "measured", isWarmup ? r + 1 : r - warmups + 1,
                    c, perCall, Double.toString(checksum));
            System.out.flush();
            if (THREAD_TIMES && pool != null) {
                StringBuilder sb = new StringBuilder("threadtimes," + (isWarmup ? "warmup" : "measured")
                        + "," + (isWarmup ? r + 1 : r - warmups + 1));
                for (long tt : threadTime) {
                    sb.append(',').append(String.format(java.util.Locale.ROOT, "%.6f", tt / 1e9));
                }
                System.err.println(sb);
            }
        }

        if (args.length >= 8) {
            writeMatrix(args[7], C, n);
        }
        if (pool != null) {
            pool.shutdown();
        }
    }
}
