/*
 * One WORKER of the distributed multiplication: a Hazelcast member in its own JVM process.
 * It stores the map entries placed on it and runs the band tasks (BandTask) sent to it by
 * the driver. It does no work until a task arrives.
 *
 * Usage: java DistMember <clusterName>
 *
 * Before joining the cluster it runs the same untimed JIT pre-warm as Assignment 3 (the
 * blocked kernel called 1000 times on 32 x 32 matrices), so that the kernel is compiled
 * before the first band task. It prints "member_ready <epoch ms>" when it has joined.
 */
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;

public class DistMember {

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("usage: java DistMember clusterName");
            System.exit(1);
        }
        BandTask.preWarm();
        HazelcastInstance hz = Hazelcast.newHazelcastInstance(ClusterConfig.member(args[0]));
        System.out.println("member_ready " + System.currentTimeMillis() + " "
                + hz.getCluster().getLocalMember().getAddress()
                + " availableProcessors=" + Runtime.getRuntime().availableProcessors());
        System.out.flush();
        // The member keeps running (Hazelcast threads are non-daemon) until the benchmark
        // script terminates the process.
    }
}
