"""
generate_training_data.py - Training Data Generator
====================================================
Generates synthetic PCAP files for training and testing the IDS.
Run this to quickly get started without needing real network captures.
"""

import random
import time
from scapy.all import IP, TCP, UDP, ICMP, wrpcap

print("=" * 70)
print("IDS Training Data Generator")
print("=" * 70)


def generate_normal_web_traffic(num_flows=50):
    """
    Generate normal web browsing traffic patterns.
    This represents typical HTTP/HTTPS traffic.
    """
    packets = []
    print(f"\n[1/4] Generating {num_flows} normal web flows...")
    
    for i in range(num_flows):
        # Simulate a web browsing session
        client_ip = f"192.168.1.{random.randint(10, 50)}"
        server_ip = random.choice([
            "93.184.216.34",    # example.com
            "142.250.185.46",   # google.com
            "151.101.1.140"     # reddit.com
        ])
        
        client_port = random.randint(49152, 65535)
        server_port = random.choice([80, 443])  # HTTP/HTTPS
        
        # Typical web session: 5-15 packets
        num_packets = random.randint(5, 15)
        
        for j in range(num_packets):
            # Request (small)
            pkt = IP(src=client_ip, dst=server_ip) / \
                  TCP(sport=client_port, dport=server_port, flags="PA") / \
                  ("GET / HTTP/1.1\r\n" * random.randint(1, 3))
            packets.append(pkt)
            
            # Response (larger)
            response_size = random.randint(500, 1500)
            pkt = IP(src=server_ip, dst=client_ip) / \
                  TCP(sport=server_port, dport=client_port, flags="PA") / \
                  ("X" * response_size)
            packets.append(pkt)
    
    print(f"   ✓ Generated {len(packets)} web packets")
    return packets


def generate_normal_dns_traffic(num_queries=30):
    """
    Generate normal DNS query patterns.
    """
    packets = []
    print(f"\n[2/4] Generating {num_queries} DNS queries...")
    
    dns_servers = ["8.8.8.8", "8.8.4.4", "1.1.1.1"]
    
    for i in range(num_queries):
        client_ip = f"192.168.1.{random.randint(10, 50)}"
        dns_server = random.choice(dns_servers)
        client_port = random.randint(49152, 65535)
        
        # DNS query (small UDP packet)
        pkt = IP(src=client_ip, dst=dns_server) / \
              UDP(sport=client_port, dport=53) / \
              (b"DNS_QUERY_" + bytes(str(i), 'utf-8'))
        packets.append(pkt)
        
        # DNS response
        pkt = IP(src=dns_server, dst=client_ip) / \
              UDP(sport=53, dport=client_port) / \
              (b"DNS_RESPONSE_" + bytes(str(i), 'utf-8'))
        packets.append(pkt)
    
    print(f"   ✓ Generated {len(packets)} DNS packets")
    return packets


def generate_normal_ssh_traffic(num_sessions=10):
    """
    Generate normal SSH session patterns.
    """
    packets = []
    print(f"\n[3/4] Generating {num_sessions} SSH sessions...")
    
    for i in range(num_sessions):
        client_ip = f"192.168.1.{random.randint(10, 50)}"
        server_ip = f"10.0.0.{random.randint(1, 5)}"
        client_port = random.randint(49152, 65535)
        
        # SSH session: moderate packet count, encrypted data
        num_packets = random.randint(20, 50)
        
        for j in range(num_packets):
            # Bidirectional encrypted traffic
            size = random.randint(64, 1500)
            
            pkt = IP(src=client_ip, dst=server_ip) / \
                  TCP(sport=client_port, dport=22, flags="PA") / \
                  (b"X" * size)
            packets.append(pkt)
            
            if random.random() > 0.3:  # Server responds 70% of time
                pkt = IP(src=server_ip, dst=client_ip) / \
                      TCP(sport=22, dport=client_port, flags="PA") / \
                      (b"Y" * random.randint(64, 500))
                packets.append(pkt)
    
    print(f"   ✓ Generated {len(packets)} SSH packets")
    return packets


def generate_normal_background_traffic(num_packets=100):
    """
    Generate various background traffic (ICMP, misc protocols).
    """
    packets = []
    print(f"\n[4/4] Generating {num_packets} background packets...")
    
    for i in range(num_packets):
        src_ip = f"192.168.1.{random.randint(1, 254)}"
        dst_ip = f"10.0.0.{random.randint(1, 254)}"
        
        # Mix of different traffic types
        traffic_type = random.choice(['icmp', 'tcp', 'udp'])
        
        if traffic_type == 'icmp':
            pkt = IP(src=src_ip, dst=dst_ip) / ICMP()
        elif traffic_type == 'tcp':
            sport = random.randint(1024, 65535)
            dport = random.choice([21, 22, 23, 25, 110, 143, 443, 3306])
            pkt = IP(src=src_ip, dst=dst_ip) / \
                  TCP(sport=sport, dport=dport, flags=random.choice(["S", "SA", "PA", "FA"]))
        else:  # udp
            sport = random.randint(1024, 65535)
            dport = random.choice([53, 123, 161, 500])
            pkt = IP(src=src_ip, dst=dst_ip) / UDP(sport=sport, dport=dport)
        
        packets.append(pkt)
    
    print(f"   ✓ Generated {len(packets)} background packets")
    return packets


def generate_attack_traffic():
    """
    Generate various attack patterns for testing.
    """
    packets = []
    print("\n[BONUS] Generating attack patterns for testing...")
    
    # 1. Port Scan
    print("   - Port scan pattern")
    attacker = "evil-scanner.com"
    target = "192.168.1.100"
    for port in range(1, 100):
        pkt = IP(src=attacker, dst=target) / \
              TCP(sport=54321, dport=port, flags="S")
        packets.append(pkt)
    
    # 2. DDoS-like pattern
    print("   - DDoS pattern")
    target = "192.168.1.200"
    for i in range(50):
        attacker = f"bot-{i}.evil.com"
        for j in range(20):
            pkt = IP(src=attacker, dst=target) / \
                  TCP(sport=random.randint(1024, 65535), dport=80, flags="S") / \
                  (b"X" * 1500)
            packets.append(pkt)
    
    # 3. Data Exfiltration
    print("   - Data exfiltration pattern")
    internal = "192.168.1.50"
    external = "suspicious-server.com"
    for i in range(200):
        pkt = IP(src=internal, dst=external) / \
              TCP(sport=34567, dport=443, flags="PA") / \
              (b"SENSITIVE_DATA_" * 100)
        packets.append(pkt)
    
    print(f"   ✓ Generated {len(packets)} attack packets")
    return packets


def main():
    """Main function to generate all training data."""
    
    print("\nThis script will generate synthetic network traffic for:")
    print("  1. Training the IDS (normal traffic)")
    print("  2. Testing the IDS (attack traffic)")
    print()
    
    # Generate normal traffic for training
    print("GENERATING TRAINING DATA (Normal Traffic)")
    print("-" * 70)
    
    all_normal_packets = []
    all_normal_packets.extend(generate_normal_web_traffic(50))
    all_normal_packets.extend(generate_normal_dns_traffic(30))
    all_normal_packets.extend(generate_normal_ssh_traffic(10))
    all_normal_packets.extend(generate_normal_background_traffic(100))
    
    # Shuffle to simulate realistic timing
    random.shuffle(all_normal_packets)
    
    # Save training data
    training_file = "training_data.pcap"
    wrpcap(training_file, all_normal_packets)
    print(f"\n✅ Training data saved to: {training_file}")
    print(f"   Total packets: {len(all_normal_packets)}")
    print(f"   Use this file to train your IDS!")
    
    # Generate attack traffic for testing
    print("\n" + "=" * 70)
    print("GENERATING TEST DATA (Attack Traffic)")
    print("-" * 70)
    
    attack_packets = generate_attack_traffic()
    
    # Save test data
    test_file = "test_attacks.pcap"
    wrpcap(test_file, attack_packets)
    print(f"\n✅ Test data saved to: {test_file}")
    print(f"   Total packets: {len(attack_packets)}")
    print(f"   Use this file to test detection!")
    
    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Generated files:")
    print(f"   1. {training_file} - Normal traffic for training")
    print(f"   2. {test_file} - Attack patterns for testing")
    print(f"\n🚀 Next steps:")
    print(f"   1. Start the IDS: python app.py")
    print(f"   2. Open browser: http://localhost:5000")
    print(f"   3. Train with: {training_file}")
    print(f"   4. Test with: {test_file}")
    print("\n💡 Tip: The system should detect the attacks but not the training data!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("\n❌ Error: Scapy not installed!")
        print("   Install it with: pip install scapy")
        print("   On Linux: sudo apt-get install python3-scapy")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure you have write permissions in this directory")
