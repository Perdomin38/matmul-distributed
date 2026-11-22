/*
 * Hazelcast settings shared by the worker members (DistMember) and the driver
 * (DistMatMul, a Hazelcast client). Everything runs on ONE machine over the loopback
 * interface (127.0.0.1); see the README for this limitation.
 *
 *   - discovery: TCP/IP on 127.0.0.1, ports 5701-5720 (listed one by one); multicast and auto-detection off
 *   - cluster name: given per configuration by the benchmark script, so members of an
 *     earlier configuration can never join a later one
 *   - the three maps (mm-A, mm-B, mm-C) keep NO backup copies (backup-count 0): each
 *     entry exists once, so the data transfers are exactly those described in the report
 *   - phone-home (usage statistics sent to Hazelcast) is disabled
 */
import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.config.Config;
import com.hazelcast.config.JoinConfig;
import com.hazelcast.config.MapConfig;
import com.hazelcast.config.NetworkConfig;

public final class ClusterConfig {

    static final String HOST = "127.0.0.1";
    static final int BASE_PORT = 5701;
    static final int PORT_COUNT = 20;
    static final String MAP_A = "mm-A";
    static final String MAP_B = "mm-B";
    static final String MAP_C = "mm-C";
    static final String EXECUTOR = "mm-exec";

    private ClusterConfig() {
    }

    static void commonProperties(java.util.function.BiConsumer<String, String> set) {
        set.accept("hazelcast.phone.home.enabled", "false");
        set.accept("hazelcast.logging.type", "jdk");
    }

    static Config member(String clusterName) {
        Config cfg = new Config();
        cfg.setClusterName(clusterName);
        commonProperties(cfg::setProperty);
        NetworkConfig net = cfg.getNetworkConfig();
        net.setPort(BASE_PORT).setPortAutoIncrement(true).setPortCount(PORT_COUNT);
        net.getInterfaces().setEnabled(true).addInterface(HOST);
        JoinConfig join = net.getJoin();
        join.getMulticastConfig().setEnabled(false);
        join.getAutoDetectionConfig().setEnabled(false);
        join.getTcpIpConfig().setEnabled(true);
        for (int port = BASE_PORT; port < BASE_PORT + PORT_COUNT; port++) {
            join.getTcpIpConfig().addMember(HOST + ":" + port);
        }
        cfg.addMapConfig(new MapConfig("mm-*").setBackupCount(0).setAsyncBackupCount(0));
        // One band task per member at a time; a small pool is enough.
        cfg.getExecutorConfig(EXECUTOR).setPoolSize(4);
        return cfg;
    }

    static ClientConfig client(String clusterName) {
        ClientConfig cfg = new ClientConfig();
        cfg.setClusterName(clusterName);
        commonProperties(cfg::setProperty);
        for (int port = BASE_PORT; port < BASE_PORT + PORT_COUNT; port++) {
            cfg.getNetworkConfig().addAddress(HOST + ":" + port);
        }
        cfg.getConnectionStrategyConfig().getConnectionRetryConfig().setClusterConnectTimeoutMillis(120_000);
        return cfg;
    }
}
