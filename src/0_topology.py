from mininet.node import Controller, OVSKernelSwitch
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.node import AccessPoint
from mn_wifi.cli import CLI
from mn_wifi.link import wlan
import sys


def create_topology():
    """Emulates an edge-IoT network with 50 wireless stations, 5 APs, and 1 edge gateway."""
    net = Mininet_wifi(controller=Controller, accessPoint=AccessPoint,
                       switch=OVSKernelSwitch, link=wlan)

    info("*** Creating network nodes\n")
    edge = net.addStation('edge', ip='10.0.0.1/8', position='50,50,0')

    aps = []
    for i in range(5):
        aps.append(net.addAccessPoint(
            f'ap{i+1}', ssid=f'ap-ssid-{i+1}',
            mode='g', channel='1', position=f'{20*(i+1)},{20*(i+1)},0'
        ))

    iot_nodes = []
    for i in range(50):
        iot_nodes.append(net.addStation(
            f'iot{i+1}', ip=f'10.0.1.{i+1}/8',
            position=f'{i+10},{i+5},0'
        ))

    info("*** Configuring SDN controller\n")
    c0 = net.addController('c0')

    info("*** Configuring wireless links\n")
    net.configureWifiNodes()

    info("*** Starting network emulation\n")
    net.build()
    c0.start()
    for ap in aps:
        ap.start([c0])

    info("*** Launching Mininet-WiFi CLI\n")
    CLI(net)

    info("*** Stopping network emulation\n")
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    if sys.platform != 'linux':
        info("Note: Mininet-WiFi requires a Linux kernel with wireless extensions.\n")
    else:
        create_topology()
